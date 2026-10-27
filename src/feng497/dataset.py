"""Eğitim öncesinde senaryo kimliklerini, grupları ve sayısal veriyi doğrula."""

import numpy as np
import pandas as pd

from .paths import DATA_FILE, INPUT_COLUMNS, TEST_SPLITS


def validate_cases(cases):
    numeric = ["D", "ex", "ey", *INPUT_COLUMNS, "kt_global"]
    required = {"case", "split", *numeric}
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"Eksik veri sütunları: {sorted(missing)}")
    if cases["case"].isna().any() or cases["case"].duplicated().any():
        raise ValueError("Senaryo kimlikleri eksiksiz ve benzersiz olmalı.")
    if cases["split"].isna().any() or set(cases["split"]) != {"train", *TEST_SPLITS}:
        raise ValueError(f"Beklenen gruplar: train, {', '.join(TEST_SPLITS)}")
    if cases.duplicated(list(INPUT_COLUMNS)).any():
        raise ValueError("Aynı girdilere sahip tekrarlanan senaryo bulundu.")
    if not np.isfinite(cases[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Girdiler ve hedefler eksiksiz ve sonlu sayılar olmalı.")
    if (cases["D"] <= 0).any() or (cases["kt_global"] <= 0).any():
        raise ValueError("Delik çapları ve maksimum gerilmeler pozitif olmalı.")
    for physical, normalized in [("D", "dw"), ("ex", "exw"), ("ey", "eyw")]:
        if not np.allclose(cases[physical] / 100.0, cases[normalized], rtol=0, atol=1e-10):
            raise ValueError(f"Birim dönüşümü tutarsız: {physical} / 100 != {normalized}")


def load_cases(path=DATA_FILE):
    cases = pd.read_csv(path)
    validate_cases(cases)
    return cases
