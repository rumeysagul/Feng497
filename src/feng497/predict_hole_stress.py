"""Delik etrafındaki teğet ve fiber ekseni gerilmelerini MPa cinsinden ver."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .hole_stress import THETA_DEG, fibre_stresses, load_model, predict_unit_hoop


def main(argv=None):
    parser = argparse.ArgumentParser(prog="feng497 predict-hole-stress", description=(
        "T300/5208 kompozit delik kenarında 360 noktalı doğrusal elastik gerilme dağılımı."
    ))
    parser.add_argument("--cap", type=float, required=True, help="Delik çapı [mm], 10–80")
    parser.add_argument("--ex", type=float, required=True, help="Yatay merkez kayması [mm], -50–50")
    parser.add_argument("--ey", type=float, required=True, help="Dikey merkez kayması [mm], -20–20")
    parser.add_argument("--fib", "--fiber-angle", type=float, required=True, help="Fiber açısı [derece], 0–90")
    parser.add_argument("--sigma-x", type=float, required=True, help="Nominal yatay gerilme [MPa]")
    parser.add_argument("--sigma-y", type=float, required=True, help="Nominal dikey gerilme [MPa]")
    parser.add_argument("--output", type=Path, help="360 noktayı içeren isteğe bağlı CSV yolu.")
    args = parser.parse_args(argv)
    if not np.isfinite([args.sigma_x, args.sigma_y]).all():
        parser.error("Nominal gerilmeler sonlu olmalı.")
    raw = pd.DataFrame([{"D": args.cap, "ex": args.ex, "ey": args.ey, "fib": args.fib}])
    try:
        x, y = predict_unit_hoop(load_model(), raw)[:, 0]
        with np.errstate(over="ignore", invalid="ignore"):
            hoop = args.sigma_x * x + args.sigma_y * y
        if not np.isfinite(hoop).all():
            raise ValueError("Uygulanan gerilme hesaplanabilir sayısal aralığı aşıyor.")
        s11, s22, t12 = fibre_stresses(hoop, args.fib)
        if args.output is not None:
            pd.DataFrame({
                "theta_deg": THETA_DEG, "stt_MPa": hoop, "s11_MPa": s11, "s22_MPa": s22, "t12_MPa": t12
            }).to_csv(args.output, index=False)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    low, high = np.argmin(hoop), np.argmax(hoop)
    print(f"Tahmini teğet gerilme aralığı: {hoop[low]:.3g} MPa ({THETA_DEG[low]}°) "
          f"ile {hoop[high]:.3g} MPa ({THETA_DEG[high]}°).")
    if args.output is not None:
        print(f"360 noktanın profili: {args.output}")


if __name__ == "__main__":
    main()
