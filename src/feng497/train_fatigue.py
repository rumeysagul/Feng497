"""S–N ilk çatlak eğrilerini öğren ve çevrim tahminini ayrılmış plakalarda ölç."""

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from .composite import TEST_SPLITS
from .fatigue import (
    DATASET_VERSION, GEOMETRY, INPUT_COLUMNS, LOAD_PATHS, LOG_CYCLES, METADATA_FILE,
    MODEL_FILE, MODELS_DIR, REPORTS_DIR, STRESS_RATIO, TARGET, life_at_peaks,
    load_training_data, make_features,
)
from .modeling import make_gpr

# Test yükleri geometriye, hedef eğrisine ve tahminlere göre seçilmez.
TEST_PEAKS_MPA = np.geomspace(1.0, 100.0, 61)


def fit_curve_model(features, targets):
    gp = make_gpr(n_features=len(INPUT_COLUMNS))
    anchor = int(np.argmin(np.abs(LOG_CYCLES - 4.5)))
    # Eğrinin ortasındaki eşikten sadece eğitim geometrileriyle çekirdeği öğren.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.set_params(gaussianprocessregressor__n_restarts_optimizer=0)
        gp.fit(features, targets[:, anchor])
    kernel = gp.named_steps["gaussianprocessregressor"].kernel_
    # 37 eşiği birlikte öğrenirken aynı çekirdeği kullan.
    model = clone(gp).set_params(gaussianprocessregressor__kernel=kernel,
                                 gaussianprocessregressor__optimizer=None)
    return model.fit(features, targets), [str(w.message) for w in caught]


def score_curves(cases, path, actual_curves, predicted_curves, baseline_curves=None):
    tables = []
    for index, (case, reference, predicted) in enumerate(zip(
        cases.itertuples(index=False), actual_curves, predicted_curves
    )):
        actual_log, actual_status = life_at_peaks(reference, TEST_PEAKS_MPA)
        predicted_log, predicted_status = life_at_peaks(predicted, TEST_PEAKS_MPA)
        table = pd.DataFrame({
            "case": case.case, "split": case.split, "fib_deg": case.fib, "yukleme": path,
            "tepe_MPa": TEST_PEAKS_MPA,
            "referans_log10_cevrim": actual_log,
            "tahmin_log10_cevrim": predicted_log,
        })
        if baseline_curves is not None:
            table["lineer_log10_cevrim"], _ = life_at_peaks(baseline_curves[index], TEST_PEAKS_MPA)
        table["referans_aralik"] = actual_status
        table["tahmin_aralik"] = predicted_status
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def metrics(rows):
    inside = rows[rows["referans_aralik"] == "aralik_ici"]
    actual = inside["referans_log10_cevrim"].to_numpy()
    error = np.abs(inside["tahmin_log10_cevrim"].to_numpy() - actual)
    result = {
        "plaka_sayisi": rows["case"].nunique(),
        "tum_yuk_testleri": len(rows),
        "referans_aralik_ici_yuk_testleri": len(inside),
        "aralik_sinifi_dogruluk_yuzde": 100 * (rows["referans_aralik"] == rows["tahmin_aralik"]).mean(),
        "referans_ici_tahmin_aralik_ici_yuzde": 100 * inside["tahmin_aralik"].eq("aralik_ici").mean(),
        "MAE_log10_cevrim": error.mean(),
        "medyan_hata_kati": 10**np.median(error),
        "P90_hata_kati": 10**np.quantile(error, 0.9),
    }
    if "lineer_log10_cevrim" in inside:
        result["lineer_MAE_log10_cevrim"] = np.abs(inside["lineer_log10_cevrim"].to_numpy() - actual).mean()
    return result


def make_report(predictions):
    splits = {split: predictions[predictions["split"] == split] for split in TEST_SPLITS}
    principal = splits["test_0_90"]
    groups = {
        "tum_testler": predictions, **splits,
        "test_0": principal[principal["fib_deg"] == 0],
        "test_90": principal[principal["fib_deg"] == 90],
    }
    by_group = [{"grup": name, "yukleme": "tumu", **metrics(rows)} for name, rows in groups.items()]
    by_path = [{"grup": split, "yukleme": path, **metrics(rows)}
               for (split, path), rows in predictions.groupby(["split", "yukleme"])]
    return pd.DataFrame(by_group + by_path)


def publish_model(models, metadata, features, expected):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODELS_DIR) as temp:
        model_path = Path(temp) / MODEL_FILE.name
        metadata_path = Path(temp) / METADATA_FILE.name
        joblib.dump(models, model_path, compress=3)
        reloaded = joblib.load(model_path)
        with threadpool_limits(limits=1):
            for path, model in reloaded.items():
                np.testing.assert_allclose(model.predict(features), expected[path], rtol=1e-12, atol=1e-12)
        metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model_path.replace(MODEL_FILE)
        metadata_path.replace(METADATA_FILE)


def main(argv=None):
    parser = argparse.ArgumentParser(description="S–N verisiyle kompozit ilk yorulma çatlağı modelini eğit.")
    parser.add_argument("--data-dir", type=Path, required=True, help="İçinde data/ ve fatigue/ olan paket dizini.")
    args = parser.parse_args(argv)
    try:
        cases, references, paths = load_training_data(args.data_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

    train = cases["split"].eq("train").to_numpy()
    test = cases["split"].isin(TEST_SPLITS).to_numpy()
    features = make_features(cases)
    models, baseline_models, warnings_by_path = {}, {}, {}
    # Dört yüklemenin tamamında aynı plaka ayrımı korunur.
    for path, curves in references.items():
        print(f"{path}: {train.sum()} plaka ve {len(LOG_CYCLES)} S–N eşiği ile eğitim.", flush=True)
        with threadpool_limits(limits=1):
            models[path], warnings_by_path[path] = fit_curve_model(features.loc[train], curves[train])
            baseline_models[path] = LinearRegression().fit(features.loc[train], curves[train])

    tables, predicted_curves = [], {}
    for path, model in models.items():
        with threadpool_limits(limits=1):
            predicted_curves[path] = model.predict(features.loc[test])
            baseline_curves = baseline_models[path].predict(features.loc[test])
        tables.append(score_curves(
            cases.loc[test, ["case", "split", *GEOMETRY]], path, references[path][test],
            predicted_curves[path], baseline_curves,
        ))
    predictions = pd.concat(tables, ignore_index=True)
    results = make_report(predictions)
    metadata = {
        "veri_surumu": DATASET_VERSION,
        "hedef": TARGET,
        "girdiler_sirayla": list(INPUT_COLUMNS),
        "gerilme_orani_R": STRESS_RATIO,
        "yukleme_yollari": {k: list(v) for k, v in LOAD_PATHS.items()},
        "log10_cevrimler": LOG_CYCLES.tolist(),
        "egitim_plaka_kimlikleri": cases.loc[train, "case"].tolist(),
        "test_plaka_sayilari": cases.loc[test, "split"].value_counts().to_dict(),
        "test_tepe_gerilmeleri_MPa": TEST_PEAKS_MPA.tolist(),
        "veri_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "yontem": "Yük başına StandardScaler + Matérn 5/2 GPR; log10 tepe MPa eğrisini tersine çevir.",
        "hedef_kaynagi": "T300/5208 doğrusal elastik ANSYS birim yük gerilmeleri + "
                          "AS4/3501-6'dan ölçeklenen S–N eğrileri + delik kenarında Hashin–Rotem.",
        "egitim_uyarilari": warnings_by_path,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "egitim_zamani_utc": datetime.now(timezone.utc).isoformat(),
    }
    publish_model(models, metadata, features.loc[test], predicted_curves)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "test_cevrim_sonuclari.csv", index=False)
    # Rapor satırları tek tek yükleri gösterir; sınır dışındaki log değerleri kırpılmıştır.
    predictions.to_csv(REPORTS_DIR / "test_cevrim_tahminleri.csv", index=False, float_format="%.8g")
    columns = ["grup", "plaka_sayisi", "MAE_log10_cevrim", "medyan_hata_kati", "P90_hata_kati"]
    print(results.loc[results["yukleme"] == "tumu", columns].round(3).to_string(index=False))
    print(f"Model: {MODEL_FILE}\nRapor: {REPORTS_DIR / 'test_cevrim_sonuclari.csv'}")


if __name__ == "__main__":
    main()
