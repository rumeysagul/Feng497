"""İlk ders: CSV'yi oku, eğitim verisini seç ve hedefi MPa'ya çevir."""

import pandas as pd

from .paths import DATA_FILE, INPUT_COLUMNS


def main():
    veri = pd.read_csv(DATA_FILE)

    print("Toplam simülasyon sayısı:", len(veri))
    print("\nHazır veri grupları:")
    print(veri["split"].value_counts().to_string())

    # Test gruplarını ayırıyoruz. İncelemeyi eğitim grubunda sürdürüyoruz.
    egitim = veri.loc[veri["split"] == "train"].copy()

    # README: kt_global = plakadaki maksimum von Mises / 100 MPa.
    # Bu dönüşüm paketteki sabit nominal yük içindir.
    egitim["max_von_mises_MPa"] = 100.0 * egitim["kt_global"]

    # X: tahmin yaparken bilinen tasarım girdileri; y: öğrenilecek sonuç.
    girdiler = list(INPUT_COLUMNS)
    hedef = "max_von_mises_MPa"
    X = egitim[girdiler]
    y = egitim[hedef]

    print("\nİlk 5 eğitim örneği:")
    print(egitim[["case"] + girdiler + [hedef]].head().to_string(index=False))
    print("\nGirdi ve hedeflerdeki eksik değer sayıları:")
    print(egitim[girdiler + [hedef]].isna().sum().to_string())
    print("\nX boyutu (satır, sütun):", X.shape)
    print("y boyutu (satır,):", y.shape)
    print("\nEğitim hedefinin özeti [MPa]:")
    print(y.describe().round(3).to_string())

    # Bu adım model eğitmez ve orijinal veri dosyalarını değiştirmez.


if __name__ == "__main__":
    main()
