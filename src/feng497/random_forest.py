"""Dördüncü ders: aynı doğrulama bölmeleriyle üç modeli karşılaştır."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from .dataset import load_cases
from .paths import INPUT_COLUMNS, REPORTS_DIR


def main():
    CIKTI = REPORTS_DIR / "04_random_forest"
    CIKTI.mkdir(parents=True, exist_ok=True)
    veri = load_cases()
    egitim = veri.loc[veri["split"] == "train"].copy()
    girdiler = list(INPUT_COLUMNS)
    X = egitim[girdiler]
    y = 100.0 * egitim["kt_global"]

    # Üçüncü dersteki bölmeleri aynen kullanıyoruz.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    modeller = {
        "Ortalama referansı": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        # Önceden belirlenmiş başlangıç ayarları; henüz ayar araması yapmıyoruz.
        # Her orman 300 ağaçtan oluşur. Her ağaç, o turdaki öğrenme grubundan
        # yerine koyarak seçilen örneklerle eğitilir. Tahminler ortalanır.
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42,
                                              n_jobs=1),
    }
    sonuclar = []
    tahminler = pd.DataFrame({"case": egitim["case"], "gercek_MPa": y})

    for ad, model in modeller.items():
        tahmin = cross_val_predict(model, X, y, cv=cv)
        tahminler[ad] = tahmin
        yuzde_hata = 100.0 * np.abs((y - tahmin) / y)
        sonuclar.append({
            "model": ad,
            "MAE_MPa": mean_absolute_error(y, tahmin),
            "RMSE_MPa": np.sqrt(mean_squared_error(y, tahmin)),
            "R2": r2_score(y, tahmin),
            "ortalama_mutlak_yuzde_hata": np.mean(yuzde_hata),
            "maksimum_mutlak_yuzde_hata": np.max(yuzde_hata),
        })

    rapor = pd.DataFrame(sonuclar)
    rapor.to_csv(CIKTI / "cv_sonuclari.csv", index=False)
    tahminler.to_csv(CIKTI / "cv_tahminleri.csv", index=False)
    print(f"{len(egitim)} eğitim örneği, 4 girdi, aynı 5 doğrulama bölmesi.")
    print(rapor.round(3).to_string(index=False))

    # En zor örnekleri gizlemiyoruz; en büyük mutlak RF hatalarını kaydediyoruz.
    hatalar = egitim[["case"] + girdiler].copy()
    hatalar["gercek_MPa"] = y
    hatalar["tahmin_MPa"] = tahminler["Random Forest"]
    hatalar["mutlak_hata_MPa"] = np.abs(y - hatalar["tahmin_MPa"])
    hatalar["mutlak_yuzde_hata"] = 100 * hatalar["mutlak_hata_MPa"] / y
    hatalar.sort_values("mutlak_hata_MPa", ascending=False).to_csv(
        CIKTI / "rf_hatalari.csv", index=False)

    adlar = ["Linear Regression", "Random Forest"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    sinirlar = [min(y.min(), tahminler[adlar].min().min()),
                max(y.max(), tahminler[adlar].max().max())]
    for ax, ad in zip(axes, adlar):
        ax.scatter(y, tahminler[ad], s=22, alpha=0.65, color="#2878a5")
        ax.plot(sinirlar, sinirlar, color="#555555", linestyle="--")
        ax.set_title(ad)
        ax.set_xlabel("Gerçek maksimum von Mises [MPa]")
        ax.set_ylabel("Doğrulama tahmini [MPa]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.suptitle("5 katlı çapraz doğrulama — test grupları kullanılmadı")
    fig.tight_layout()
    fig.savefig(CIKTI / "model_karsilastirma.png", dpi=160)
    plt.close(fig)

    print("\nEn büyük 5 mutlak Random Forest hatası:")
    print(hatalar.sort_values("mutlak_hata_MPa", ascending=False).head()
          .round(3).to_string(index=False))
    print("\nDosyalar:", CIKTI)
    print("Bunlar geliştirme sonuçlarıdır; nihai test başarısı değildir.")


if __name__ == "__main__":
    main()
