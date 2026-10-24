"""Yedinci ders: mm cinsinden girdilerle kaydedilmiş modelden tahmin al."""

import argparse
import json
import math
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .paths import MODELS_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "300x100x2 mm çelik plakada maksimum von Mises tahmini. "
        "Sabit sigma_x=100 MPa, sigma_y=lam*100 MPa."))
    parser.add_argument("--cap", type=float, required=True, help="Delik çapı [mm]")
    parser.add_argument("--ex", type=float, required=True,
                        help="Plaka merkezinden yatay kayma [mm]; sağ pozitif")
    parser.add_argument("--ey", type=float, required=True,
                        help="Plaka merkezinden dikey kayma [mm]; yukarı pozitif")
    parser.add_argument("--lam", type=float, required=True, help="sigma_y / sigma_x")
    args = parser.parse_args(argv)

    if not all(math.isfinite(v) for v in [args.cap, args.ex, args.ey, args.lam]):
        parser.error("Tüm girdiler sonlu sayılar olmalı.")
    if args.cap <= 0:
        parser.error("Delik çapı sıfırdan büyük olmalı.")
    # Merkez (150+ex, 50+ey). Delik plakanın içinde kalmalı.
    yatay_bosluk = 150.0 - abs(args.ex) - args.cap / 2.0
    dikey_bosluk = 50.0 - abs(args.ey) - args.cap / 2.0
    if min(yatay_bosluk, dikey_bosluk) <= 0:
        parser.error("Delik plaka sınırına değiyor veya dışına taşıyor.")

    klasor = MODELS_DIR
    model_yolu = klasor / "maksimum_von_mises_gpr.joblib"
    bilgi_yolu = klasor / "model_bilgileri.json"
    if not model_yolu.exists() or not bilgi_yolu.exists():
        parser.error("Model bulunamadı. Önce python -m feng497 train çalıştırılmalı.")
    bilgiler = json.loads(bilgi_yolu.read_text(encoding="utf-8"))
    # Fiziksel uzunluklar plaka genişliği W=100 mm'ye bölünür.
    degerler = {"dw": args.cap / 100.0, "exw": args.ex / 100.0,
                "eyw": args.ey / 100.0, "lam": args.lam}
    X_yeni = pd.DataFrame([degerler], columns=bilgiler["girdiler_sirayla"])
    aralik_disi = []
    for sutun, deger in degerler.items():
        aralik = bilgiler["girdi_egitim_araliklari"][sutun]
        if not aralik["min"] <= deger <= aralik["max"]:
            aralik_disi.append(
                f"{sutun}={deger:g}; eğitim aralığı "
                f"[{aralik['min']:g}, {aralik['max']:g}]")
    for mesaj in aralik_disi:
        print("Eğitim aralığı dışında:", mesaj)
    if dikey_bosluk < 10.0:
        print("Veri kapsamı dışında: delik ile üst/alt kenar arasında 10 mm'den az var.")
    if aralik_disi or dikey_bosluk < 10.0:
        print("Bu koşulda tahmin ekstrapolasyondur; bildirilen test hataları garanti değildir.")
    else:
        print("Her girdi eğitimdeki tekil min–max aralığında; bu bir hata garantisi değildir.")

    # Kendi ürettiğimiz yerel model dosyası: scaler ve log/exp dönüşümü içerir.
    model = joblib.load(model_yolu)
    with threadpool_limits(limits=1):
        tahmin = float(model.predict(X_yeni)[0])
    if not np.isfinite(tahmin) or tahmin <= 0:
        raise ValueError("Model geçerli bir pozitif tahmin üretmedi.")
    print("\nPlaka: 300 x 100 x 2 mm; doğrusal elastik çelik")
    print(f"Delik çapı: {args.cap:g} mm; merkez: "
          f"({150 + args.ex:g}, {50 + args.ey:g}) mm")
    print(f"Uygulanan sigma_x: 100 MPa; sigma_y: {100 * args.lam:g} MPa")
    print("Model girdileri:", degerler)
    print(f"\nTahmini maksimum von Mises gerilmesi: {tahmin:.2f} MPa")
    print("Bu bir ML tahminidir; yeni bir ANSYS çözümü veya hasar tahmini değildir.")


if __name__ == "__main__":
    main()
