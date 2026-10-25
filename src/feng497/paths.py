"""Çalışma dizininden bağımsız veri, model ve rapor yolları."""

import os
from pathlib import Path

# Editable kurulumda src/feng497/ dosyalarının iki üstü depo köküdür.
# Başka kurulumlarda veri deposu FENG497_ROOT ile gösterilebilir.
PROJECT_ROOT = Path(
    os.environ.get("FENG497_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
DATASET_VERSION = "v2"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / f"plate_hole_dataset_{DATASET_VERSION}"
DATA_FILE = DATA_DIR / "data" / "cases.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_COLUMNS = ("dw", "exw", "eyw", "lam")
TEST_SPLITS = ("test_interp", "test_mid", "test_big")
TEST_LABELS = {
    "test_interp": "Küçük delikler (10–60 mm)",
    "test_mid": "Orta delikler (60–65 mm)",
    "test_big": "Büyük delikler (60–80 mm)",
}
