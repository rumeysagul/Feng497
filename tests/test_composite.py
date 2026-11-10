"""Kompozit hedef kapsamı, eğitim/test ayrımı ve kayıtlı model entegrasyonu."""

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
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from feng497.composite import (
    DATA_FILE, INPUT_COLUMNS, METADATA_FILE, MODEL_FILE, RAW_INPUTS, REPORTS_DIR,
    TARGET, TEST_SPLITS, load_cases, make_features, target_values, validate_cases,
)


class CompositeDataTests(unittest.TestCase):
    def test_features_use_geometry_and_distinguish_principal_fiber_directions(self):
        raw = pd.DataFrame([
            {"D": 30, "ex": 10, "ey": 5, "lam": 0, "fib": 0},
            {"D": 30, "ex": 10, "ey": 5, "lam": 0, "fib": 90},
        ])
        X = make_features(raw)
        np.testing.assert_allclose(X.iloc[0], [.3, .1, .05, 0, 1, 0, .3, .4, .3, .7, 1.25], atol=1e-12)
        np.testing.assert_allclose(X.iloc[1], [.3, .1, .05, 0, -1, 0, .3, .4, .3, .7, 1.25], atol=1e-12)
        self.assertEqual(X.columns.tolist(), list(INPUT_COLUMNS))

    def test_fe_results_never_enter_features_and_global_target_is_not_hole_target(self):
        cases = load_cases()
        contaminated = cases.copy()
        for column in contaminated.columns.difference(list(RAW_INPUTS)):
            contaminated[column] = 999999
        pd.testing.assert_frame_equal(make_features(cases), make_features(contaminated))
        off_hole = cases.loc[cases["tw_R_global"] < cases["tw_R"] - 1e-10]
        self.assertGreater(len(off_hole), 0)
        self.assertTrue((target_values(off_hole) < off_hole["fail_load_tw_mpa"]).all())

    def test_invalid_composite_data_is_rejected(self):
        cases = load_cases()
        for issue in ("missing", "duplicate_id", "duplicate_inputs", "split", "angle",
                      "target", "global_target", "hole_target", "units", "ligament"):
            with self.subTest(issue=issue):
                invalid = cases.copy()
                if issue == "missing":
                    invalid.loc[0, "fib"] = np.nan
                elif issue == "duplicate_id":
                    invalid.loc[1, "case"] = invalid.loc[0, "case"]
                elif issue == "duplicate_inputs":
                    invalid.loc[1, list(RAW_INPUTS)] = invalid.loc[0, list(RAW_INPUTS)].values
                elif issue == "split":
                    invalid.loc[0, "split"] = "test_big"
                elif issue == "angle":
                    invalid.loc[0, "fib"] = 91
                elif issue == "target":
                    invalid.loc[0, "tw_R_global"] = 0
                elif issue == "global_target":
                    invalid.loc[0, "tw_R_global"] = invalid.loc[0, "tw_R"] + 1
                elif issue == "hole_target":
                    invalid.loc[0, "fail_load_tw_mpa"] *= 2
                elif issue == "units":
                    invalid.loc[0, "dw"] *= 100
                elif issue == "ligament":
                    invalid.loc[0, ["D", "ey"]] = [80, 1]
                with self.assertRaises(ValueError):
                    validate_cases(invalid)

    def test_principal_test_contains_both_angles_only(self):
        cases = load_cases()
        self.assertEqual(len(cases), 808)
        self.assertEqual(cases.groupby("split").size().to_dict(),
                         {"train": 648, "test_interp": 100, "test_0_90": 60})
        principal = cases.loc[cases["split"] == "test_0_90"]
        self.assertEqual(principal.groupby("fib").size().to_dict(), {0.0: 30, 90.0: 30})


class CompositeModelTests(unittest.TestCase):
    def test_saved_predictions_match_all_held_out_cases(self):
        cases = load_cases()
        reference = pd.read_csv(REPORTS_DIR / "test_tahminleri.csv")
        expected = cases.loc[cases["split"].isin(TEST_SPLITS)]
        self.assertEqual(set(reference["case"]), set(expected["case"]))
        self.assertEqual(len(reference), len(expected))
        self.assertFalse(reference["case"].duplicated().any())
        ordered = cases.set_index("case").loc[reference["case"]]
        self.assertEqual(ordered["split"].tolist(), reference["split"].tolist())
        model = joblib.load(MODEL_FILE)
        with threadpool_limits(limits=1):
            predicted = model.predict(make_features(ordered))
        np.testing.assert_allclose(predicted, reference["tahmin_MPa"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(target_values(ordered), reference["gercek_MPa"])
        self.assertTrue((predicted > 0).all())
        info = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        self.assertEqual(info["hedef"], TARGET)
        self.assertEqual(info["veri_sha256"], hashlib.sha256(DATA_FILE.read_bytes()).hexdigest())
        self.assertEqual(info["model_sha256"], hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest())

    def test_scaler_and_gpr_use_only_training_rows_and_global_log_target(self):
        cases = load_cases()
        train = cases.loc[cases["split"] == "train"]
        info = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        self.assertEqual(info["egitim_case_ids"], train["case"].tolist())
        self.assertEqual(info["egitim_ornek_sayisi"], len(train))
        self.assertEqual(info["girdiler_sirayla"], list(INPUT_COLUMNS))
        model = joblib.load(MODEL_FILE)
        scaler = model.regressor_.named_steps["standardscaler"]
        gpr = model.regressor_.named_steps["gaussianprocessregressor"]
        self.assertEqual(scaler.n_samples_seen_, len(train))
        np.testing.assert_allclose(gpr.X_train_, scaler.transform(make_features(train)), atol=1e-12)
        log_y = np.log(target_values(train).to_numpy())
        np.testing.assert_allclose(gpr._y_train_mean, log_y.mean())
        np.testing.assert_allclose(gpr._y_train_std, log_y.std())
        np.testing.assert_allclose(gpr.y_train_, (log_y - log_y.mean()) / log_y.std(), atol=1e-12)
        cv = pd.read_csv(REPORTS_DIR / "cv_tahminleri.csv")
        self.assertEqual(cv["case"].tolist(), train["case"].tolist())
        expected_folds = np.zeros(len(train), dtype=int)
        for number, (_, validation) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), 1):
            expected_folds[validation] = number
        np.testing.assert_array_equal(cv["fold"], expected_folds)
        self.assertFalse(set(cv["case"]) & set(cases.loc[cases["split"] != "train", "case"]))

    def test_test_metrics_and_principal_subgroups_match_predictions(self):
        predictions = pd.read_csv(REPORTS_DIR / "test_tahminleri.csv")
        results = pd.read_csv(REPORTS_DIR / "test_sonuclari.csv").set_index("split")
        principal = pd.read_csv(REPORTS_DIR / "test_0_90_sonuclari.csv").set_index("fib_deg")
        for split in TEST_SPLITS:
            subset = predictions.loc[predictions["split"] == split]
            error = 100 * np.abs(subset["gercek_MPa"] - subset["tahmin_MPa"]) / subset["gercek_MPa"]
            self.assertEqual(results.loc[split, "n"], len(subset))
            self.assertAlmostEqual(results.loc[split, "ortalama_mutlak_yuzde_hata"], error.mean())
        for angle in (0, 90):
            subset = predictions.loc[(predictions["split"] == "test_0_90") & (predictions["fib"] == angle)]
            self.assertEqual(principal.loc[angle, "n"], len(subset))
            self.assertAlmostEqual(principal.loc[angle, "ortalama_mutlak_yuzde_hata"],
                                   subset["mutlak_yuzde_hata"].mean())

    def test_prediction_cli_from_another_directory(self):
        reference = pd.read_csv(REPORTS_DIR / "test_tahminleri.csv")
        row = reference.loc[(reference["split"] == "test_0_90") & (reference["fib"] == 90)].iloc[0]
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        environment.pop("FENG497_ROOT", None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, "-m", "feng497", "predict-composite",
                "--cap", str(row["D"]), "--ex", str(row["ex"]), "--ey", str(row["ey"]),
                "--lam", str(row["lam"]), "--fib", str(row["fib"]),
            ], cwd=directory, env=environment, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{row['tahmin_MPa']:.2f} MPa", result.stdout)
        self.assertIn("T300/5208", result.stdout)

    def test_cli_help_and_imports_work_without_data_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "FENG497_ROOT": directory, "PYTHONIOENCODING": "utf-8"}
            for args in (["-m", "feng497", "predict-composite", "--help"],
                         ["-c", "import feng497.composite; import feng497.train_composite; "
                          "import feng497.predict_composite"]):
                result = subprocess.run([sys.executable, *args], cwd=directory, env=env,
                                        capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prediction_cli_rejects_invalid_angles_and_ligaments(self):
        for cap, ey, angle in [(30, 0, "nan"), (30, 0, "91"), (80, 1, "0")]:
            with self.subTest(cap=cap, ey=ey, angle=angle):
                result = subprocess.run([
                    sys.executable, "-m", "feng497", "predict-composite", "--cap", str(cap),
                    "--ex", "0", "--ey", str(ey), "--lam", "0", "--fib", angle,
                ], capture_output=True, text=True, encoding="utf-8",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
