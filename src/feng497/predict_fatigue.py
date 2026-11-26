"""Sabit genlikli dört yükleme biçimi için ilk yorulma çatlağı çevrim tahmini."""

import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .fatigue import (
    INPUT_COLUMNS, LOAD_PATHS, LOG_CYCLES, METADATA_FILE, MODEL_FILE, STRESS_RATIO,
    TARGET, life_at_peaks, make_features,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "T300/5208 tek tabakada delik kenarında ilk yorulma çatlağı. "
        "Sabit genlikli yükleme, R=0.1. Fiber açısı +x ekseninden ölçülür."
    ))
    parser.add_argument("--cap", type=float, required=True, help="Delik çapı [mm], 10–80")
    parser.add_argument("--ex", type=float, required=True, help="Yatay merkez kayması [mm], -50–50")
    parser.add_argument("--ey", type=float, required=True, help="Dikey merkez kayması [mm], -20–20")
    parser.add_argument("--fib", "--fiber-angle", type=float, required=True, help="Fiber açısı [derece], 0–90")
    parser.add_argument("--load-path", choices=LOAD_PATHS, default="x",
                        help="x: yatay; y: dikey; biaxial: eşit iki eksen; shear: zıt işaretli iki eksen.")
    parser.add_argument("--peak-mpa", type=float, required=True,
                        help="Nominal sigma_x tepe gerilmesi [MPa]; y için sigma_y. Pozitif değer.")
    args = parser.parse_args(argv)
    try:
        features = make_features(pd.DataFrame([{"D": args.cap, "ex": args.ex, "ey": args.ey, "fib": args.fib}]))
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.peak_mpa) or args.peak_mpa <= 0:
        parser.error("Tepe gerilmesi pozitif ve sonlu olmalı.")
    if not MODEL_FILE.is_file() or not METADATA_FILE.is_file():
        parser.error("Yorulma modeli bulunamadı. Önce train-fatigue --data-dir <paket> komutunu çalıştırın.")
    try:
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        expected = {
            "hedef": TARGET, "girdiler_sirayla": list(INPUT_COLUMNS), "gerilme_orani_R": STRESS_RATIO,
            "yukleme_yollari": {k: list(v) for k, v in LOAD_PATHS.items()}, "log10_cevrimler": LOG_CYCLES.tolist(),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Model kapsamı veya girdi sırası yorulma koduyla eşleşmiyor.")
        if hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest() != metadata.get("model_sha256"):
            raise ValueError("Model dosyası kayıtlı hash ile eşleşmiyor.")
        model = joblib.load(MODEL_FILE)[args.load_path]
        with threadpool_limits(limits=1):
            curve = model.predict(features)[0]
        log_cycles, status = life_at_peaks(curve, args.peak_mpa)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Yükleme: {args.load_path}; R={STRESS_RATIO:g}; nominal tepe gerilmesi: {args.peak_mpa:g} MPa")
    if status == "ilk_cevrim":
        print("Tahmin: ilk çevrim içinde. Yük, modelin 1 çevrim eşiğini aşıyor.")
    elif status == "aralik_ustu":
        print(f"Tahmin: 10^{LOG_CYCLES[-1]:g} çevrimin üzerinde (model aralığının üst sınırı).")
    else:
        print(f"Delik kenarında tahmini ilk yorulma çatlağı: {10**float(log_cycles):.3g} çevrim.")
    print("Referans: AS4/3501-6'dan uyarlanan S–N eğrileri ve ANSYS birim yük sonuçları.")


if __name__ == "__main__":
    main()
