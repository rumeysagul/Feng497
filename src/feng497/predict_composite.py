"""Beş fiziksel girdiyle plaka genelinde kompozit ilk hasar yükü tahmini."""

import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .composite import INPUT_COLUMNS, METADATA_FILE, MODEL_FILE, TARGET, make_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "300x100x2 mm T300/5208 tek yönlü kompozitte plaka genelinde "
        "Tsai-Wu ilk hasar yükü. Fiber açısı +x ekseninden ölçülür."))
    parser.add_argument("--cap", type=float, required=True, help="Delik çapı [mm], 10–80")
    parser.add_argument("--ex", type=float, required=True, help="Yatay merkez kayması [mm], -50–50")
    parser.add_argument("--ey", type=float, required=True, help="Dikey merkez kayması [mm], -20–20")
    parser.add_argument("--lam", type=float, required=True, help="sigma_y / sigma_x, -1–1")
    parser.add_argument("--fib", "--fiber-angle", type=float, required=True,
                        help="Fiber açısı [derece], +x ekseninden 0–90")
    args = parser.parse_args(argv)
    raw = pd.DataFrame([{"D": args.cap, "ex": args.ex, "ey": args.ey,
                         "lam": args.lam, "fib": args.fib}])
    try:
        features = make_features(raw)
    except ValueError as error:
        parser.error(str(error))
    if not MODEL_FILE.exists() or not METADATA_FILE.exists():
        parser.error("Kompozit model bulunamadı. Önce python -m feng497 train-composite çalıştırın.")
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    if metadata["girdiler_sirayla"] != list(INPUT_COLUMNS) or metadata["hedef"] != TARGET:
        parser.error("Model kapsamı veya girdi sırası bu kompozit koduyla eşleşmiyor.")
    if hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest() != metadata["model_sha256"]:
        parser.error("Model dosyası kayıtlı hash ile eşleşmiyor.")
    model = joblib.load(MODEL_FILE)
    with threadpool_limits(limits=1):
        predicted = float(model.predict(features)[0])
    if not np.isfinite(predicted) or predicted <= 0:
        raise ValueError("Geçersiz ilk hasar yükü tahmini.")
    print("Malzeme: T300/5208 karbon/epoksi, tek yönlü tek tabaka")
    print(f"Delik: {args.cap:g} mm; merkez: ({150 + args.ex:g}, {50 + args.ey:g}) mm")
    print(f"Fiber açısı: {args.fib:g}° (+x ekseninden); sigma_y/sigma_x: {args.lam:g}")
    print(f"\nTahmini plaka geneli ilk hasar yükü (sigma_x): {predicted:.2f} MPa")
    print(f"Aynı yüklemede sigma_y: {args.lam * predicted:.2f} MPa")
    print(f"100 MPa referansına göre yük çarpanı: {predicted / 100.0:.6f}")
    print("Tsai-Wu ilk hasar ölçütüne ait ML tahminidir. Çatlak boyutu veya ilerlemesi hesaplanmaz.")
    print("Girdiler veri kapsamındadır; bildirilen test hataları yeni senaryolar için garanti değildir.")


if __name__ == "__main__":
    main()
