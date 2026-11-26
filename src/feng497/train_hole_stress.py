"""Delik kenarı gerilmesini öğren ve tahminleri ilk çatlak hesabında değerlendir."""

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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .composite import TEST_SPLITS
from .fatigue import (
    DATASET_VERSION, GEOMETRY, LOAD_PATHS, load_sn_strengths, load_training_data, load_unit_profiles,
    log_peaks_from_profiles, make_features,
)
from .hole_stress import (
    AXES, COMPONENTS, METADATA_FILE, MODEL_DIR, MODEL_FILE, MODEL_SCOPE, REPORT_DIR, fibre_stresses,
    predict_unit_hoop, shift_profiles,
)
from .modeling import make_gpr
from .train_fatigue import make_report, score_curves


def fit_profiles(features, profiles):
    # Tam alanı öğren. Çekirdek optimizasyonu için önde gelen bileşenler yeterli.
    leading = PCA(n_components=4, svd_solver="randomized", random_state=42).fit_transform(profiles)
    gp = make_gpr(n_features=features.shape[1])
    gp.set_params(gaussianprocessregressor__n_restarts_optimizer=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.fit(features, leading)
    kernel = gp.named_steps["gaussianprocessregressor"].kernel_
    model = clone(gp).set_params(gaussianprocessregressor__kernel=kernel,
                                 gaussianprocessregressor__optimizer=None)
    return model.fit(features, profiles), [str(w.message) for w in caught]


def relative_profile_error(reference, predicted):
    return np.linalg.norm(predicted - reference, axis=1) / np.linalg.norm(reference, axis=1)


def select_models(cases, hoops, train):
    features = make_features(cases)
    angles = cases["fib"].to_numpy()
    train_ids = np.flatnonzero(train)
    orientation = np.where(np.isin(angles[train], (0, 90)), angles[train], -1)
    learning, validation = train_test_split(train_ids, test_size=0.2, random_state=42, stratify=orientation)
    models, frames, training_info = {}, {}, {}
    for axis, hoop in zip(AXES, hoops):
        responses = {"global": hoop, "fibre": shift_profiles(hoop, angles)}
        validation_errors = {}
        for frame, targets in responses.items():
            print(f"{axis}/{frame}: {len(learning)} plaka ile eğitim içi karşılaştırma.", flush=True)
            model, _ = fit_profiles(features.iloc[learning], targets[learning])
            predicted = model.predict(features.iloc[validation])
            if frame == "fibre":
                predicted = shift_profiles(predicted, -angles[validation])
            validation_errors[frame] = float(relative_profile_error(hoop[validation], predicted).mean())
        frame = min(validation_errors, key=validation_errors.get)
        print(f"{axis}/{frame}: seçildi, {len(train_ids)} plaka ile son eğitim.", flush=True)
        models[axis], caught = fit_profiles(features.iloc[train_ids], responses[frame][train_ids])
        frames[axis] = frame
        training_info[axis] = {"dogrulama_bagil_L2": validation_errors, "uyarilar": caught}
    return models, frames, training_info


def stress_report(cases, reference, predicted):
    groups = {
        "tum_testler": np.ones(len(cases), dtype=bool),
        **{split: cases["split"].eq(split).to_numpy() for split in TEST_SPLITS},
        **{f"test_{angle}": (cases["split"].eq("test_0_90") & cases["fib"].eq(angle)).to_numpy()
           for angle in (0, 90)},
    }
    rows = []
    for group, mask in groups.items():
        for axis, axis_reference, axis_predicted in zip(AXES, reference, predicted):
            for component, ref, pred in zip(("stt", *COMPONENTS), axis_reference, axis_predicted):
                ref, pred = ref[mask], pred[mask]
                profile_error = relative_profile_error(ref, pred)
                ref_max = np.abs(ref).max(axis=1)
                peak_error = np.abs(np.abs(pred).max(axis=1) - ref_max) / ref_max
                rows.append({
                    "grup": group, "birim_yuk": axis, "bilesen": component,
                    "plaka_sayisi": len(ref),
                    "medyan_profil_bagil_L2_yuzde": 100 * np.median(profile_error),
                    "P90_profil_bagil_L2_yuzde": 100 * np.quantile(profile_error, 0.9),
                    "medyan_tepe_mutlak_yuzde_hata": 100 * np.median(peak_error),
                    "RMSE_100_MPa_yukte_MPa": 100 * np.sqrt(np.mean((pred - ref)**2)),
                })
    return pd.DataFrame(rows)


def save_model(bundle, info, cases, expected_hoops):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODEL_DIR) as temp:
        model_path = Path(temp) / MODEL_FILE.name
        meta_path = Path(temp) / METADATA_FILE.name
        joblib.dump(bundle, model_path, compress=3)
        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(predict_unit_hoop(reloaded, cases), expected_hoops, rtol=1e-12, atol=1e-12)
        info["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model_path.replace(MODEL_FILE)
        meta_path.replace(METADATA_FILE)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="feng497 train-hole-stress",
                                     description="Delik kenarında 360 noktadaki gerilme profillerini öğren.")
    parser.add_argument("--data-dir", type=Path, required=True, help="İçinde data/ ve fatigue/ olan paket dizini.")
    args = parser.parse_args(argv)
    try:
        cases, reference_curves, paths = load_training_data(args.data_dir)
        fields = load_unit_profiles(paths["profiles"], cases["case"], ("stt", *COMPONENTS))
        strengths = load_sn_strengths(paths["curves"])
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

    train = cases["split"].eq("train").to_numpy()
    test = cases["split"].isin(TEST_SPLITS).to_numpy()
    hoops = fields[:, 0]
    with threadpool_limits(limits=1):
        models, frames, training_info = select_models(cases, hoops, train)
    bundle = {"models": models, "frames": frames, "strengths": strengths}
    test_cases = cases.loc[test, ["case", "split", *GEOMETRY]]
    predicted_hoops = predict_unit_hoop(bundle, test_cases)
    predicted_stresses = fibre_stresses(predicted_hoops, test_cases["fib"].to_numpy()).swapaxes(1, 2)
    all_predicted = np.concatenate((predicted_hoops[:, None], predicted_stresses), axis=1)
    scores = stress_report(test_cases, fields[:, :, test], all_predicted)

    life_rows = pd.concat([
        score_curves(test_cases, path, reference_curves[path][test],
                     log_peaks_from_profiles(predicted_stresses, strengths, weights))
        for path, weights in LOAD_PATHS.items()
    ], ignore_index=True)
    life_report = make_report(life_rows)
    info = {
        **MODEL_SCOPE,
        "veri_surumu": DATASET_VERSION,
        "yontem": "Matérn 5/2 GPR, 360 teğet gerilme noktası; serbest kenarda fiber ekseni dönüşümü",
        "yorulma_referansi": "ANSYS birim yük gerilmeleri ve AS4/3501-6'dan uyarlanan S–N eğrileri",
        "egitim_plaka_kimlikleri": cases.loc[train, "case"].tolist(),
        "test_plaka_sayilari": cases.loc[test, "split"].value_counts().to_dict(),
        "aci_koordinatlari": frames,
        "egitim_dogrulama": training_info,
        "veri_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "egitim_zamani_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_model(bundle, info, test_cases, predicted_hoops)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(REPORT_DIR / "test_gerilme_sonuclari.csv", index=False)
    life_report.to_csv(REPORT_DIR / "test_cevrim_sonuclari.csv", index=False)
    life_rows.to_csv(REPORT_DIR / "test_cevrim_tahminleri.csv", index=False, float_format="%.8g")
    print(scores.loc[(scores["grup"] == "tum_testler") & (scores["bilesen"] == "stt")].round(3).to_string(index=False))
    columns = ["grup", "plaka_sayisi", "MAE_log10_cevrim", "medyan_hata_kati", "P90_hata_kati"]
    print(life_report.loc[life_report["yukleme"] == "tumu", columns].round(3).to_string(index=False))
    print(f"Model: {MODEL_FILE}", flush=True)


if __name__ == "__main__":
    main()
