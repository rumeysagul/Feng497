"""Serbest delik kenarında birim yük profilleri ve yorulmaya giden gerilmeler."""

import hashlib
import io
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .fatigue import INPUT_COLUMNS, LOAD_PATHS, LOG_CYCLES, STRESS_RATIO, log_peaks_from_profiles, make_features
from .paths import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models" / "hole_stress_v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "hole_stress_v1"
MODEL_FILE = MODEL_DIR / "delik_kenari_gerilme_gpr.joblib"
METADATA_FILE = MODEL_DIR / "model_bilgileri.json"
THETA_DEG = np.arange(-180, 180)
COMPONENTS = ("s11", "s22", "t12")
AXES = ("X", "Y")
MODEL_SCOPE = {
    "hedef": "birim_eksenel_yuk_basina_delik_kenari_teget_gerilmesi",
    "girdiler_sirayla": list(INPUT_COLUMNS),
    "theta_deg": THETA_DEG.tolist(),
    "eksenler": list(AXES),
    "gerilme_orani_R": STRESS_RATIO,
    "log10_cevrimler": LOG_CYCLES.tolist(),
}


def shift_profiles(profiles, offsets_deg):
    """Periyodik profili theta yerine theta + offset noktalarında değerlendir."""
    profiles = np.asarray(profiles)
    spectrum = np.fft.rfft(profiles, axis=-1)
    frequency = np.fft.rfftfreq(profiles.shape[-1])
    phase = np.exp(2j * np.pi * np.asarray(offsets_deg)[..., None] * frequency)
    # Çift uzunluklu gerçek FFT'de tek başına kalan Nyquist bileşenini koru.
    if profiles.shape[-1] % 2 == 0:
        phase[..., -1] = 1
    return np.fft.irfft(spectrum * phase, n=profiles.shape[-1], axis=-1)


def fibre_stresses(hoop, fibre_deg):
    """Serbest kenardaki teğet gerilmeyi (..., bileşen, açı) biçimine çevir."""
    relative = np.deg2rad(THETA_DEG - np.asarray(fibre_deg)[..., None])
    s, c = np.sin(relative), np.cos(relative)
    return np.stack((hoop * s**2, hoop * c**2, -hoop * s * c), axis=-2)


def predict_unit_hoop(bundle, cases):
    """Çıktı eksenleri iki birim yük, plaka ve 360 delik açısıdır."""
    features = make_features(cases)
    fibre_deg = cases["fib"].to_numpy()
    hoops = []
    with threadpool_limits(limits=1):
        for axis in AXES:
            predicted = bundle["models"][axis].predict(features)
            if bundle["frames"][axis] == "fibre":
                predicted = shift_profiles(predicted, -fibre_deg)
            hoops.append(predicted)
    result = np.array(hoops)
    if result.shape != (len(AXES), len(cases), len(THETA_DEG)) or not np.isfinite(result).all():
        raise ValueError("Model geçerli bir delik kenarı gerilme profili üretmedi.")
    return result


def fatigue_curve(bundle, cases, load_path):
    hoops = predict_unit_hoop(bundle, cases)
    stresses = fibre_stresses(hoops, cases["fib"].to_numpy()).swapaxes(1, 2)
    return log_peaks_from_profiles(stresses, bundle["strengths"], LOAD_PATHS[load_path])


def load_model():
    if not MODEL_FILE.is_file() or not METADATA_FILE.is_file():
        raise ValueError("Gerilme modeli bulunamadı. Önce train-hole-stress --data-dir <paket> çalıştırın.")
    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    if any(meta.get(key) != value for key, value in MODEL_SCOPE.items()):
        raise ValueError("Gerilme modelinin kapsamı veya girdi sırası eşleşmiyor.")
    stored = MODEL_FILE.read_bytes()
    if hashlib.sha256(stored).hexdigest() != meta.get("model_sha256"):
        raise ValueError("Gerilme modeli kayıtlı dosya hash'iyle eşleşmiyor.")
    return joblib.load(io.BytesIO(stored))
