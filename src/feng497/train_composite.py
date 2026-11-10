"""Kompozit modelini eğitim içi CV ile denetle, sonra ayrılmış testleri değerlendir."""

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .composite import (
    DATA_FILE, DATASET_VERSION, INPUT_COLUMNS, METADATA_FILE, MODEL_FILE,
    MODELS_DIR, RAW_INPUTS, REPORTS_DIR, TARGET, TARGET_LABEL, TEST_SPLITS,
    load_cases, make_features, target_values,
)
from .modeling import make_gpr
from .paths import PROJECT_ROOT


def metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    percentage = 100.0 * np.abs(actual - predicted) / actual
    return {
        "n": len(actual),
        "MAE_MPa": float(mean_absolute_error(actual, predicted)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "ortalama_mutlak_yuzde_hata": float(np.mean(percentage)),
        "maksimum_mutlak_yuzde_hata": float(np.max(percentage)),
        "P95_mutlak_yuzde_hata": float(np.percentile(percentage, 95)),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    cases = load_cases()
    train = cases.loc[cases["split"] == "train"].copy()
    X = make_features(train)
    y = target_values(train)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Mimari ve hedef test tahminleri hesaplanmadan önce sabittir.
    # Çelik sonuçlarıyla kompozit hedefleri birleştirilmez.
    config = {
        "veri_surumu": DATASET_VERSION,
        "hedef": TARGET,
        "girdiler_sirayla": list(INPUT_COLUMNS),
        "yontem": "StandardScaler + GPR Matérn 5/2 ARD, log/exp hedef",
        "n_restarts_optimizer": 3,
        "model_random_state": 0,
        "cv_n_splits": 5,
        "cv_shuffle": True,
        "cv_random_state": 42,
        "secim": "Önceden sabit log GPR; doğrusal regresyon yalnızca CV referansı. "
                 "Test grupları model veya özellik seçimine katılmıyor.",
    }
    (REPORTS_DIR / "egitim_ayarlari.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    fold_numbers = np.zeros(len(train), dtype=int)
    for number, (_, validation) in enumerate(folds, 1):
        fold_numbers[validation] = number
    cv_predictions = pd.DataFrame({
        "case": train["case"].to_numpy(), "fold": fold_numbers,
        "gercek_MPa": y.to_numpy(),
    })
    cv_results = []
    all_warnings = []
    models = {
        "Linear Regression": LinearRegression(),
        "GPR (log hedef)": make_gpr(log_target=True, n_features=len(INPUT_COLUMNS)),
    }
    for name, estimator in models.items():
        oof = np.empty(len(train))
        for number, (learning, validation) in enumerate(folds, 1):
            print(f"{name}: CV {number}/5, {len(learning)} öğrenme / "
                  f"{len(validation)} doğrulama", flush=True)
            fold_model = clone(estimator)
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fold_model.fit(X.iloc[learning], y.iloc[learning])
                oof[validation] = fold_model.predict(X.iloc[validation])
            all_warnings.extend(
                f"{name} CV {number}: {w.category.__name__}: {w.message}" for w in caught)
        if not np.isfinite(oof).all():
            raise ValueError(f"{name}: sonlu olmayan CV tahmini.")
        cv_predictions[name] = oof
        cv_results.append({"model": name, **metrics(y, oof)})
    pd.DataFrame(cv_results).to_csv(REPORTS_DIR / "cv_sonuclari.csv", index=False)
    cv_predictions.to_csv(REPORTS_DIR / "cv_tahminleri.csv", index=False)

    model = clone(models["GPR (log hedef)"])
    print(f"Son kompozit modeli {len(train)} eğitim senaryosuyla eğitiliyor...", flush=True)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    all_warnings.extend(f"Son eğitim: {w.category.__name__}: {w.message}" for w in caught)
    joblib.dump(model, MODEL_FILE)
    reloaded = joblib.load(MODEL_FILE)

    # Ayrılmış testler yalnızca bu aşamada tahmin için kullanılır.
    predictions = []
    results = []
    for split in TEST_SPLITS:
        test = cases.loc[cases["split"] == split].copy()
        features = make_features(test)
        actual = target_values(test)
        with threadpool_limits(limits=1):
            predicted = model.predict(features)
            reloaded_prediction = reloaded.predict(features)
        np.testing.assert_allclose(predicted, reloaded_prediction, rtol=1e-12, atol=1e-12)
        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise ValueError("Kompozit model geçerli pozitif yük tahmini üretmedi.")
        table = test[["case", "split", *RAW_INPUTS]].copy()
        table["gercek_MPa"] = actual
        table["tahmin_MPa"] = predicted
        table["mutlak_hata_MPa"] = np.abs(actual - predicted)
        table["mutlak_yuzde_hata"] = 100.0 * table["mutlak_hata_MPa"] / actual
        table["delik_kenari_ilk_hasar_MPa"] = test["fail_load_tw_mpa"]
        table["global_ilk_hasar_delik_kenarinda"] = test["tw_global_on_hole"]
        predictions.append(table)
        results.append({"split": split, **metrics(actual, predicted)})
    predictions = pd.concat(predictions, ignore_index=True)
    results = pd.DataFrame(results)
    predictions.to_csv(REPORTS_DIR / "test_tahminleri.csv", index=False)
    results.to_csv(REPORTS_DIR / "test_sonuclari.csv", index=False)
    principal_results = []
    for angle in (0, 90):
        subset = predictions.loc[
            (predictions["split"] == "test_0_90") & (predictions["fib"] == angle)]
        principal_results.append({"fib_deg": angle,
                                  **metrics(subset["gercek_MPa"], subset["tahmin_MPa"])})
    principal_results = pd.DataFrame(principal_results)
    principal_results.to_csv(REPORTS_DIR / "test_0_90_sonuclari.csv", index=False)

    metadata = {
        **config,
        "egitim_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "veri_yolu": DATA_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "veri_sha256": sha256(DATA_FILE),
        "model_sha256": sha256(MODEL_FILE),
        "hedef_birimi": "MPa",
        "hedef_kapsami": "Plakanın herhangi bir yerinde Tsai-Wu ilk tabaka hasarı; "
                        "sabit lambda ile birlikte ölçeklenen yükte nominal sigma_x.",
        "delik_hedefinden_farkli_senaryo": int((cases["tw_R"] - cases["tw_R_global"] > 1e-10).sum()),
        "egitim_ornek_sayisi": len(train),
        "egitim_case_ids": train["case"].tolist(),
        "test_ornek_sayilari": {s: int((cases["split"] == s).sum()) for s in TEST_SPLITS},
        "fiziksel_egitim_araliklari": {
            s: {"min": float(train[s].min()), "max": float(train[s].max())} for s in RAW_INPUTS},
        "girdi_egitim_araliklari": {
            s: {"min": float(X[s].min()), "max": float(X[s].max())} for s in INPUT_COLUMNS},
        "malzeme": {
            "ad": "T300/5208 karbon/epoksi UD tek tabaka",
            "E1_MPa": 181000, "E2_MPa": 10300, "G12_MPa": 7170, "nu12": 0.28,
            "XT_MPa": 1500, "XC_MPa": 1500, "YT_MPa": 40, "YC_MPa": 246, "S_MPa": 68,
            "F12": "-0.5 * sqrt(F11 * F22), sağlayıcının varsayımı",
        },
        "kapsam": "300x100x2 mm delikli plaka, 2D statik, plane stress, "
                  "doğrusal elastik ortotropik tek tabaka; fiber açısı +x ekseninden 0–90 derece.",
        "sinirlar": "İlk hasar ölçütü tahminidir; fiziksel deney doğrulaması, çatlak boyutu, "
                    "çatlak ilerlemesi, nihai taşıma kapasitesi veya hasar konumu tahmini değildir. "
                    "Von Mises hedefi kullanılmaz. Çelik örnekleri eğitime eklenmez.",
        "ogrenilen_kernel": str(model.regressor_.named_steps["gaussianprocessregressor"].kernel_),
        "yeniden_yukleme_kontrolu": f"{len(predictions)} test tahmini, rtol=atol=1e-12",
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
        "kaynak_kod_sha256": {
            f"src/feng497/{name}": sha256(PROJECT_ROOT / "src" / "feng497" / name)
            for name in ("composite.py", "modeling.py", "train_composite.py")
        },
        "sure_saniye": round(time.perf_counter() - started, 3),
    }
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "egitim_uyarilari.txt").write_text(
        "\n".join(all_warnings) if all_warnings else "Eğitim uyarısı yok.\n", encoding="utf-8")

    groups = [
        ("Ara açılar", predictions.loc[predictions["split"] == "test_interp"]),
        ("Fiber açısı 0°", predictions.loc[
            (predictions["split"] == "test_0_90") & (predictions["fib"] == 0)]),
        ("Fiber açısı 90°", predictions.loc[
            (predictions["split"] == "test_0_90") & (predictions["fib"] == 90)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, table) in zip(axes, groups):
        limits = [0, 1.05 * max(table["gercek_MPa"].max(), table["tahmin_MPa"].max())]
        ax.scatter(table["gercek_MPa"], table["tahmin_MPa"], s=24, alpha=0.8)
        ax.plot(limits, limits, "--", color="#555555")
        ax.set(xlabel="FE ilk hasar yükü [MPa]", ylabel="ML tahmini [MPa]",
               title=f"{title} ({len(table)} senaryo)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.suptitle(TARGET_LABEL)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "test_gercek_ve_tahmin.png", dpi=160)
    plt.close(fig)
    print("\nEğitim içi 5 katlı çapraz doğrulama:")
    print(pd.DataFrame(cv_results).round(4).to_string(index=False))
    print("\nAyrılmış testler:")
    print(results.round(4).to_string(index=False))
    print("\n0° ve 90° ayrı sonuçlar:")
    print(principal_results.round(4).to_string(index=False))
    print(f"\nModel: {MODEL_FILE}")
    print(f"Eğitim süresi: {metadata['sure_saniye']:.1f} s; uyarı: {len(all_warnings)}")


if __name__ == "__main__":
    main()
