"""Çalışma dizininden bağımsız veri, model ve rapor yolları."""

import os
from pathlib import Path

# Editable kurulumda src/feng497/ dosyalarının iki üstü depo köküdür.
# Başka kurulumlarda veri deposu FENG497_ROOT ile gösterilebilir.
PROJECT_ROOT = Path(
    os.environ.get("FENG497_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "plate_hole_dataset_v1"
DATA_FILE = DATA_DIR / "data" / "cases.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_COLUMNS = ("dw", "exw", "eyw", "lam")
