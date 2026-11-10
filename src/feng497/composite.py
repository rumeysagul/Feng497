"""UD kompozit veri kapsamı, geometri özellikleri ve ilk hasar hedefi."""

import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT

DATASET_VERSION = "UD_composite_dataset_v1"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_VERSION
DATA_FILE = DATA_DIR / "data" / "cases.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "composite_v1"
REPORTS_DIR = PROJECT_ROOT / "reports" / "composite_v1"
MODEL_FILE = MODELS_DIR / "kompozit_ilk_hasar_gpr.joblib"
METADATA_FILE = MODELS_DIR / "model_bilgileri.json"
RAW_INPUTS = ("D", "ex", "ey", "lam", "fib")
INPUT_COLUMNS = (
    "dw", "exw", "eyw", "lam", "cos_2fib", "sin_2fib",
    "lig_top_w", "lig_bot_w", "lig_min_w", "net_w", "dist_end_w",
)
TEST_SPLITS = ("test_interp", "test_0_90")
TARGET = "100 * tw_R_global"
TARGET_LABEL = "Plaka genelinde Tsai-Wu ilk hasar yükü [MPa]"
RAW_BOUNDS = {
    "D": (10.0, 80.0), "ex": (-50.0, 50.0), "ey": (-20.0, 20.0),
    "lam": (-1.0, 1.0), "fib": (0.0, 90.0),
}


def validate_inputs(cases):
    """Veri sağlayıcısının geometri, açı ve yükleme kapsamını uygula."""
    missing = set(RAW_INPUTS).difference(cases.columns)
    if missing:
        raise ValueError(f"Eksik fiziksel girdi sütunları: {sorted(missing)}")
    if not np.isfinite(cases[list(RAW_INPUTS)].to_numpy(dtype=float)).all():
        raise ValueError("Girdiler eksiksiz ve sonlu sayılar olmalı.")
    for column, (low, high) in RAW_BOUNDS.items():
        if not cases[column].between(low, high).all():
            raise ValueError(f"{column} kapsam dışında: [{low:g}, {high:g}].")
    ligament = 50.0 - cases["D"] / 2.0 - cases["ey"].abs()
    if (ligament < 10.0 - 1e-8).any():
        raise ValueError("Delik ile üst/alt kenar arasında en az 10 mm kalmalı.")


def make_features(cases):
    """Yalnızca beş fiziksel girdiden özellik üret; FE sonuçları girdi değildir."""
    validate_inputs(cases)
    result = pd.DataFrame(index=cases.index)
    result["dw"] = cases["D"] / 100.0
    result["exw"] = cases["ex"] / 100.0
    result["eyw"] = cases["ey"] / 100.0
    result["lam"] = cases["lam"]
    angle = np.radians(2.0 * cases["fib"])
    result["cos_2fib"] = np.cos(angle)
    result["sin_2fib"] = np.sin(angle)
    result["lig_top_w"] = (50.0 - cases["D"] / 2.0 - cases["ey"]) / 100.0
    result["lig_bot_w"] = (50.0 - cases["D"] / 2.0 + cases["ey"]) / 100.0
    result["lig_min_w"] = result[["lig_top_w", "lig_bot_w"]].min(axis=1)
    result["net_w"] = (100.0 - cases["D"]) / 100.0
    result["dist_end_w"] = (150.0 - cases["ex"].abs() - cases["D"] / 2.0) / 100.0
    return result[list(INPUT_COLUMNS)]


def target_values(cases):
    """Yük oranı sabitken tüm plakada ilk hasarın başladığı sigma_x [MPa]."""
    return 100.0 * cases["tw_R_global"]


def validate_cases(cases):
    normalized = (
        "dw", "exw", "eyw", "lig_top_w", "lig_bot_w", "lig_min_w",
        "net_w", "dist_end_w",
    )
    numeric = [*RAW_INPUTS, *normalized, "tw_R", "tw_R_global", "fail_load_tw_mpa"]
    missing = {"case", "split", *numeric}.difference(cases.columns)
    if missing:
        raise ValueError(f"Eksik kompozit veri sütunları: {sorted(missing)}")
    ids = pd.to_numeric(cases["case"], errors="coerce")
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError("Senaryo kimlikleri pozitif tam sayılar olmalı.")
    if cases["case"].duplicated().any():
        raise ValueError("Senaryo kimlikleri benzersiz olmalı.")
    if set(cases["split"]) != {"train", *TEST_SPLITS}:
        raise ValueError("Beklenen gruplar: train, test_interp, test_0_90.")
    if (cases["split"] == "train").sum() < 5:
        raise ValueError("Beş katlı doğrulama için en az beş eğitim senaryosu gerekli.")
    if not np.isfinite(cases[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Kompozit girdileri ve hedefleri sonlu sayılar olmalı.")
    if cases.duplicated(list(RAW_INPUTS)).any():
        raise ValueError("Aynı girdili senaryolar eğitim ve testte tekrarlanamaz.")
    features = make_features(cases)
    for column in normalized:
        if not np.allclose(cases[column], features[column], rtol=0, atol=1e-8):
            raise ValueError(f"Birim veya geometri dönüşümü tutarsız: {column}.")
    if (cases[["tw_R", "tw_R_global", "fail_load_tw_mpa"]] <= 0).any().any():
        raise ValueError("İlk hasar oranları ve yükleri pozitif olmalı.")
    if not np.allclose(cases["fail_load_tw_mpa"], 100.0 * cases["tw_R"]):
        raise ValueError("Delik kenarı yükü 100 * tw_R ile eşleşmiyor.")
    if (cases["tw_R_global"] > cases["tw_R"] + 1e-10).any():
        raise ValueError("Global ilk hasar yükü delik kenarı yükünü aşamaz.")
    principal = cases.loc[cases["split"] == "test_0_90", "fib"]
    if set(principal) != {0.0, 90.0}:
        raise ValueError("test_0_90 yalnızca 0 ve 90 derece örneklerini içermeli.")


def load_cases(path=DATA_FILE):
    cases = pd.read_csv(path)
    validate_cases(cases)
    return cases


def main():
    cases = load_cases()
    print("Malzeme: T300/5208 karbon/epoksi, tek yönlü tek tabaka, ortotropik")
    print("Hedef:", TARGET, "[MPa]; plakanın herhangi bir yerinde ilk hasar")
    print(cases.groupby("split").size().rename("senaryo").to_string())
    principal = cases.loc[cases["split"] == "test_0_90"]
    print("0/90 testleri:", principal.groupby("fib").size().to_dict())
    print("Global hedefin delik kenarından farklı olduğu senaryo:",
          int((cases["tw_R"] - cases["tw_R_global"] > 1e-10).sum()))
    print("Modelin 11 girdisi:", ", ".join(INPUT_COLUMNS))


if __name__ == "__main__":
    main()
