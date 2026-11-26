"""Delik kenarında ilk yorulma çatlağının gerilme eşikleri ve çevrim ömrü."""

import numpy as np
import pandas as pd

from .composite import INPUT_COLUMNS as COMPOSITE_COLUMNS
from .composite import make_features as make_composite_features
from .paths import PROJECT_ROOT

DATASET_VERSION = "UD_composite_unitloads_v1"
MODELS_DIR = PROJECT_ROOT / "models" / "fatigue_v1"
REPORTS_DIR = PROJECT_ROOT / "reports" / "fatigue_v1"
MODEL_FILE = MODELS_DIR / "kompozit_yorulma_gpr.joblib"
METADATA_FILE = MODELS_DIR / "model_bilgileri.json"
GEOMETRY = ("D", "ex", "ey", "fib")
INPUT_COLUMNS = tuple(column for column in COMPOSITE_COLUMNS if column != "lam")
LOAD_PATHS = {
    "x": (1, 0),
    "y": (0, 1),
    "biaxial": (1, 1),
    "shear": (1, -1),
}
STRESS_RATIO = 0.1
LOG_CYCLES = np.arange(0.0, 9.25, 0.25)
TARGET = "delik_kenari_ilk_catlak_log10_tepe_MPa"
DATA_FILES = {
    "cases": "data/cases.csv",
    "profiles": "data/hole_profiles_unit.npz",
    "curves": "fatigue/sn_curves.csv",
    "targets": "fatigue/fatigue_targets.csv",
}
STRENGTH_MODES = (
    "fibre_tension", "fibre_compression", "transverse_tension",
    "transverse_compression", "in_plane_shear",
)


def make_features(cases):
    # Her yük biçimi için ayrı eğri öğrenilir. Eski lam_v1 burada kullanılmaz.
    return make_composite_features(cases.assign(lam=0.0))[list(INPUT_COLUMNS)]


def first_crack_peaks(s11, s22, t12, strengths):
    """Her plaka ve çevrim düzeyi için 360 delik noktası arasındaki en düşük eşik."""
    limits = []
    for xt, xc, yt, yc, shear in strengths:
        x_limit = np.where(s11 >= 0.0, xt, xc)
        y_limit = np.where(s22 >= 0.0, yt, yc)
        with np.errstate(divide="ignore"):
            fibre = x_limit / np.abs(s11)
            matrix = 1.0 / np.hypot(s22 / y_limit, t12 / shear)
        limits.append(np.minimum(fibre, matrix).min(axis=1))
    return np.column_stack(limits)


def load_training_data(data_dir):
    paths = {name: data_dir / relative for name, relative in DATA_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("Yorulma paketi eksik: " + ", ".join(missing))
    cases = pd.read_csv(paths["cases"])
    expected_splits = {"train": 648, "test_interp": 100, "test_0_90": 60}
    ids = cases["case"]
    if (
        len(cases) != sum(expected_splits.values())
        or cases["split"].value_counts().to_dict() != expected_splits
        or ids.duplicated().any()
        or not pd.api.types.is_integer_dtype(ids)
        or (ids <= 0).any()
        or cases.duplicated(list(GEOMETRY)).any()
    ):
        raise ValueError("Plaka kimlikleri, geometriler veya eğitim/test grupları geçersiz.")
    make_features(cases)
    if cases.loc[cases["split"] == "test_0_90", "fib"].value_counts().to_dict() != {0.: 30, 90.: 30}:
        raise ValueError("Uç açı testi her açı için 30 ayrı plaka içermeli.")

    targets = pd.read_csv(paths["targets"]).set_index("case", verify_integrity=True)
    if set(targets.index) != set(ids) or not np.array_equal(targets.loc[ids, "split"], cases["split"]):
        raise ValueError("Hazır hedeflerin plaka kimlikleri veya grupları eşleşmiyor.")
    targets = targets.loc[ids]
    sn = pd.read_csv(paths["curves"])
    sn = sn.loc[np.isclose(sn["r"], STRESS_RATIO), ["mode", "log10_N", "peak_MPa_T300_5208"]]
    sn = sn.pivot(index="log10_N", columns="mode", values="peak_MPa_T300_5208").sort_index()
    if not np.array_equal(sn.index, LOG_CYCLES) or set(sn.columns) != set(STRENGTH_MODES):
        raise ValueError("S–N tablosunun modları veya çevrim aralığı beklenen kapsamla eşleşmiyor.")
    strengths = sn[list(STRENGTH_MODES)].to_numpy()
    if not np.isfinite(strengths).all() or (strengths <= 0).any() or (np.diff(strengths, axis=0) >= 0).any():
        raise ValueError("S–N dayanımları pozitif olmalı ve çevrim sayısıyla azalmalı.")

    with np.load(paths["profiles"], allow_pickle=False) as profiles:
        profile_ids = pd.Index(profiles["case"])
        if not profile_ids.is_unique or set(profile_ids) != set(ids):
            raise ValueError("Gerilme profillerinin plaka kimlikleri eşleşmiyor.")
        if not np.array_equal(profiles["theta_deg"], np.arange(-180, 180)):
            raise ValueError("Delik kenarındaki 360 açının sırası değişmiş.")
        order = profile_ids.get_indexer(ids)
        x, y = [
            [profiles[f"{stress}_{axis}"][order] for stress in ("s11", "s22", "t12")]
            for axis in ("X", "Y")
        ]
    if any(a.shape != (len(cases), 360) or not np.isfinite(a).all() for a in [*x, *y]):
        raise ValueError("Delik kenarı gerilme profilleri eksik veya geçersiz.")

    log_peaks = {}
    for path, (wx, wy) in LOAD_PATHS.items():
        peaks = first_crack_peaks(*(wx * a + wy * b for a, b in zip(x, y)), strengths)
        if not np.isfinite(peaks).all() or (peaks <= 0).any() or (np.diff(peaks, axis=1) >= 0).any():
            raise ValueError(f"{path} için hesaplanan eşikler geçersiz.")
        suffix = path.upper() if path in ("x", "y") else path
        for exponent in (3, 6):
            reference = targets[f"fatigue_1e{exponent}_{suffix}_MPa"].to_numpy()
            if not np.allclose(peaks[:, int(exponent / 0.25)], reference, rtol=0, atol=0.00006):
                raise ValueError(f"{path} için 10^{exponent} çevrim eşiği sağlayıcının tablosuyla eşleşmiyor.")
        log_peaks[path] = np.log10(peaks)
    return cases, log_peaks, paths


def monotone_log_peaks(log_peaks):
    """Öğrenilen tepe gerilmesi eşiğini artmayacak şekilde sınırla."""
    log_peaks = np.asarray(log_peaks, dtype=float)
    if log_peaks.ndim == 0 or log_peaks.shape[-1] != len(LOG_CYCLES) or not np.isfinite(log_peaks).all():
        raise ValueError("Model geçerli bir S–N eğrisi üretmedi.")
    return np.minimum.accumulate(log_peaks, axis=-1)


def life_at_peaks(log_peaks, peak_mpa):
    """S–N eğrisini tersine çevir. Uçların dışındaki sonuçlara aralık etiketi ver."""
    curve = monotone_log_peaks(log_peaks)
    peaks = np.asarray(peak_mpa, dtype=float)
    if curve.ndim != 1 or not np.isfinite(peaks).all() or (peaks <= 0).any():
        raise ValueError("Tepe gerilmesi pozitif ve sonlu olmalı.")
    log_load = np.log10(peaks)
    # Düz bir bölümde aynı eşiğe ulaşılan ilk çevrim geçerlidir.
    right = np.searchsorted(-curve, -log_load, side="left")
    right = np.clip(right, 1, len(curve) - 1)
    left = right - 1
    span = curve[left] - curve[right]
    fraction = np.divide(curve[left] - log_load, span, out=np.zeros_like(log_load), where=span > 0)
    log_life = LOG_CYCLES[left] + np.clip(fraction, 0, 1) * (LOG_CYCLES[right] - LOG_CYCLES[left])
    log_life = np.where(log_load < curve[-1], LOG_CYCLES[-1], log_life)
    status = np.where(log_load > curve[0], "ilk_cevrim",
                      np.where(log_load < curve[-1], "aralik_ustu", "aralik_ici"))
    return log_life, status
