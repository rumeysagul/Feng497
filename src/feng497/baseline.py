"""Üçüncü ders: eğitim grubu içinde 5 katlı çapraz doğrulama."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from .paths import DATA_FILE, INPUT_COLUMNS, REPORTS_DIR


def main():
    CIKTI = REPORTS_DIR / "03_ilk_model"
    CIKTI.mkdir(parents=True, exist_ok=True)
    veri = pd.read_csv(DATA_FILE)
    egitim = veri.loc[veri["split"] == "train"].copy()
    girdiler = list(INPUT_COLUMNS)
    X = egitim[girdiler]
    y = 100.0 * egitim["kt_global"]

    # Her turda 240 örnek öğrenme, 60 örnek doğrulama için kullanılır.
    # Beş tur sonunda her örnek, onu görmeyen bir modelle bir kez tahmin edilir.
    # Sabit seed, tekrar çalıştırınca aynı bölmelerin kullanılmasını sağlar.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    modeller = {
        "Ortalama referansı": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
    }
    sonuclar = []
    tahminler = pd.DataFrame({"case": egitim["case"], "gercek_MPa": y})

    for ad, model in modeller.items():
        tahmin = cross_val_predict(model, X, y, cv=cv)
        tahminler[ad] = tahmin
        sonuclar.append({
            "model": ad,
            "MAE_MPa": mean_absolute_error(y, tahmin),
            "RMSE_MPa": np.sqrt(mean_squared_error(y, tahmin)),
            "R2": r2_score(y, tahmin),
            "ortalama_mutlak_yuzde_hata": 100 * np.mean(np.abs((y - tahmin) / y)),
            "maksimum_mutlak_yuzde_hata": 100 * np.max(np.abs((y - tahmin) / y)),
            "negatif_tahmin_sayisi": int(np.sum(tahmin < 0)),
        })

    rapor = pd.DataFrame(sonuclar)
    rapor.to_csv(CIKTI / "cv_sonuclari.csv", index=False)
    tahminler.to_csv(CIKTI / "cv_tahminleri.csv", index=False)
    print("Yalnızca 300 eğitim örneği; 5 katlı çapraz doğrulama.")
    print("Ölçümler, her örneğin eğitim dışında kaldığı turdaki tahmininden hesaplandı.")
    print(rapor.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    sinirlar = [min(y.min(), tahminler[list(modeller)].min().min()),
                max(y.max(), tahminler[list(modeller)].max().max())]
    for ax, ad in zip(axes, modeller):
        ax.scatter(y, tahminler[ad], s=22, alpha=0.65, color="#2878a5")
        ax.plot(sinirlar, sinirlar, color="#555555", linestyle="--")
        ax.set_title(ad)
        ax.set_xlabel("Gerçek maksimum von Mises [MPa]")
        ax.set_ylabel("Doğrulama tahmini [MPa]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.suptitle("5 katlı çapraz doğrulama — test grupları kullanılmadı")
    fig.tight_layout()
    fig.savefig(CIKTI / "gercek_ve_tahmin.png", dpi=160)
    plt.close(fig)
    print("\nDosyalar:", CIKTI)


if __name__ == "__main__":
    main()
