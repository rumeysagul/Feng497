"""Veri, kaydedilmiş model ve terminal komutlarının tutarlılığını denetle."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from feng497.dataset import load_cases, validate_cases
from feng497.paths import (
    DATA_FILE, DATASET_VERSION, INPUT_COLUMNS, MODELS_DIR, PROJECT_ROOT,
    REPORTS_DIR, TEST_SPLITS,
)


class ProjectIntegrationTests(unittest.TestCase):
    def test_saved_model_still_matches_reference_predictions(self):
        """Yeni modelin kayıtlı tüm test tahminleri ve veri/model hash'leri eşleşmeli."""
        cases = load_cases()
        reference = pd.read_csv(REPORTS_DIR / "06_son_test/test_tahminleri.csv")
        expected = cases.loc[cases["split"].isin(TEST_SPLITS)]
        self.assertEqual(len(reference), len(expected))
        self.assertEqual(set(reference["case"]), set(expected["case"]))
        self.assertFalse(reference["case"].duplicated().any())
        ordered = cases.set_index("case").loc[reference["case"]]
        self.assertEqual(ordered["split"].tolist(), reference["split"].tolist())
        model = joblib.load(MODELS_DIR / "maksimum_von_mises_gpr.joblib")
        with threadpool_limits(limits=1):
            actual = model.predict(ordered[list(INPUT_COLUMNS)])
        np.testing.assert_allclose(actual, reference["tahmin_MPa"], rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(100 * ordered["kt_global"], reference["gercek_MPa"])
        metadata = json.loads((MODELS_DIR / "model_bilgileri.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(), metadata["veri_sha256"])
        self.assertEqual(metadata["veri_surumu"], DATASET_VERSION)
        self.assertEqual(
            hashlib.sha256((MODELS_DIR / "maksimum_von_mises_gpr.joblib").read_bytes()).hexdigest(),
            metadata["model_sha256"],
        )

    def test_model_was_fit_only_on_training_cases(self):
        """Test satırlarının scaler veya GPR öğrenmesine karışmasını yakala."""
        cases = load_cases()
        train = cases.loc[cases["split"] == "train"]
        metadata = json.loads((MODELS_DIR / "model_bilgileri.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["egitim_case_ids"], train["case"].tolist())
        self.assertEqual(metadata["egitim_ornek_sayisi"], len(train))
        self.assertEqual(metadata["girdiler_sirayla"], list(INPUT_COLUMNS))
        self.assertEqual(metadata["test_ornek_sayilari"], {
            split: int((cases["split"] == split).sum()) for split in TEST_SPLITS
        })
        model = joblib.load(MODELS_DIR / "maksimum_von_mises_gpr.joblib")
        scaler = model.regressor_.named_steps["standardscaler"]
        gpr = model.regressor_.named_steps["gaussianprocessregressor"]
        self.assertEqual(scaler.n_samples_seen_, len(train))
        np.testing.assert_allclose(
            gpr.X_train_, scaler.transform(train[list(INPUT_COLUMNS)]), rtol=1e-12, atol=1e-12,
        )

    def test_v2_keeps_v1_cases_and_their_train_test_separation(self):
        old = pd.read_csv(PROJECT_ROOT / "data/raw/plate_hole_dataset_v1/data/cases.csv").set_index("case")
        new = load_cases().set_index("case")
        shared = [column for column in old.columns if column != "split"]
        pd.testing.assert_frame_equal(old[shared], new.loc[old.index, shared], check_dtype=False, check_exact=True)
        self.assertEqual(
            old["split"].replace({"test_extrap": "test_mid"}).tolist(),
            new.loc[old.index, "split"].tolist(),
        )

    def test_invalid_data_is_rejected_before_training(self):
        cases = load_cases()
        for name in ["missing_value", "duplicate_case", "duplicate_inputs", "unknown_split", "wrong_units"]:
            with self.subTest(name=name):
                invalid = cases.copy()
                if name == "missing_value":
                    invalid.loc[0, "kt_global"] = np.nan
                elif name == "duplicate_case":
                    invalid.loc[1, "case"] = invalid.loc[0, "case"]
                elif name == "duplicate_inputs":
                    invalid.loc[1, list(INPUT_COLUMNS)] = invalid.loc[0, list(INPUT_COLUMNS)].values
                elif name == "unknown_split":
                    invalid.loc[0, "split"] = "test_extrap"
                else:
                    invalid.loc[0, "dw"] *= 100
                with self.assertRaises(ValueError):
                    validate_cases(invalid)

    def test_prediction_cli_works_outside_project_directory(self):
        """Terminalin bulunduğu klasör değişince veri/model yolu kırılmamalı."""
        environment = os.environ.copy()
        environment.pop("FENG497_ROOT", None)
        environment["PYTHONIOENCODING"] = "utf-8"
        # Fiziksel mm girdilerinin doğru normalize edilmesini büyük bir test deliğinde denetle.
        reference = pd.read_csv(REPORTS_DIR / "06_son_test/test_tahminleri.csv")
        example = reference.loc[reference["split"] == "test_big"].iloc[0]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "feng497", "predict", "--cap", str(example["D"]),
                 "--ex", str(example["ex"]), "--ey", str(example["ey"]), "--lam", str(example["lam"])],
                cwd=directory, env=environment, text=True, encoding="utf-8",
                capture_output=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{example['tahmin_MPa']:.2f} MPa", result.stdout)

    def test_imports_do_not_read_data_train_or_create_reports(self):
        """Boş veri deposunda modülleri import etmek bile eğitim başlatmamalı."""
        modules = ("dataset", "inspect_data", "plot_data", "baseline", "random_forest", "gpr", "train", "predict")
        with tempfile.TemporaryDirectory() as directory:
            environment = {**os.environ, "FENG497_ROOT": directory, "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [sys.executable, "-c", "; ".join(f"import feng497.{module}" for module in modules)],
                cwd=directory, env=environment, text=True, encoding="utf-8",
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prediction_help_does_not_require_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "feng497", "predict", "--help"],
                cwd=directory, env={**os.environ, "FENG497_ROOT": directory, "PYTHONIOENCODING": "utf-8"},
                text=True, encoding="utf-8", capture_output=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--cap", "--ex", "--ey", "--lam"):
            self.assertIn(flag, result.stdout)


if __name__ == "__main__":
    unittest.main()
