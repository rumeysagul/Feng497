"""Yapısal düzenlemenin dosya yollarını ve mevcut tahminleri koruduğunu denetle."""

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

from feng497.paths import DATA_FILE, INPUT_COLUMNS, MODELS_DIR, REPORTS_DIR


class ProjectIntegrationTests(unittest.TestCase):
    def test_saved_model_still_matches_reference_predictions(self):
        """Dosya taşıma ve ortak kod, kayıtlı modeli veya sonuç birimini bozmamalı."""
        cases = pd.read_csv(DATA_FILE)
        reference = pd.read_csv(REPORTS_DIR / "06_son_test/test_tahminleri.csv")
        ordered = cases.set_index("case").loc[reference["case"]]
        model = joblib.load(MODELS_DIR / "maksimum_von_mises_gpr.joblib")
        with threadpool_limits(limits=1):
            actual = model.predict(ordered[list(INPUT_COLUMNS)])
        np.testing.assert_allclose(actual, reference["tahmin_MPa"], rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(100 * ordered["kt_global"], reference["gercek_MPa"])
        metadata = json.loads((MODELS_DIR / "model_bilgileri.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(), metadata["veri_sha256"])

    def test_prediction_cli_works_outside_project_directory(self):
        """Terminalin bulunduğu klasör değişince veri/model yolu kırılmamalı."""
        environment = os.environ.copy()
        environment.pop("FENG497_ROOT", None)
        environment["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "feng497", "predict", "--cap", "30", "--ex", "0", "--ey", "0", "--lam", "0"],
                cwd=directory, env=environment, text=True, encoding="utf-8",
                capture_output=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("338.18 MPa", result.stdout)

    def test_imports_do_not_read_data_train_or_create_reports(self):
        """Boş veri deposunda modülleri import etmek bile eğitim başlatmamalı."""
        modules = ("inspect_data", "plot_data", "baseline", "random_forest", "gpr", "train", "predict")
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
