"""Beşinci ders: doğrudan ve log hedefli GPR'yi aynı bölmelerde karşılaştır."""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from threadpoolctl import threadpool_limits

from .dataset import load_cases
from .paths import INPUT_COLUMNS, REPORTS_DIR
from .modeling import make_gpr


def main():
    CIKTI = REPORTS_DIR / "05_gpr"
    CIKTI.mkdir(parents=True, exist_ok=True)
    veri = load_cases()
    egitim = veri.loc[veri["split"] == "train"].copy()
    girdiler = list(INPUT_COLUMNS)
    X = egitim[girdiler]
    y = 100.0 * egitim["kt_global"]

    # Hem scaler hem GPR yalnızca o turun öğrenme örneklerine fit edilir.
    # Log dönüşümü, MPa hedefi pozitif olduğu için kullanılabilir.
    # Geri dönüşüm predict içinde yapılır; tüm metrikler MPa üzerinden hesaplanır.
    modeller = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42,
                                              n_jobs=1),
        "GPR": make_gpr(),
        "GPR (log hedef)": make_gpr(log_target=True),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    tahminler = pd.DataFrame({"case": egitim["case"], "gercek_MPa": y})
    sonuclar = []
    uyarilar = []

    for ad, model in modeller.items():
        print("Çalışıyor:", ad, flush=True)
        # Küçük matrislerde gereksiz paralel BLAS iş yükünü önler.
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as kayit:
            warnings.simplefilter("always", ConvergenceWarning)
            tahmin = cross_val_predict(model, X, y, cv=cv)
        for uyari in kayit:
            uyarilar.append(f"{ad}: {uyari.category.__name__}: {uyari.message}")
        if not np.isfinite(tahmin).all():
            raise ValueError(f"{ad}: sonlu olmayan tahmin bulundu.")
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
    (CIKTI / "egitim_uyarilari.txt").write_text(
        "\n".join(uyarilar) if uyarilar else "Eğitim uyarısı yok.\n", encoding="utf-8")
    print(f"\n{len(egitim)} eğitim örneği, aynı 4 girdi ve aynı 5 doğrulama bölmesi.")
    print(rapor.round(3).to_string(index=False))
    print("Eğitim uyarısı sayısı:", len(uyarilar), "(ayrıntılar dosyada)")

    # İki GPR yaklaşımının hatalarını ayrıca inceleyebilmek için kaydet.
    for ad, dosya in [("GPR", "gpr_hatalari.csv"),
                      ("GPR (log hedef)", "gpr_log_hatalari.csv")]:
        hatalar = egitim[["case"] + girdiler].copy()
        hatalar["gercek_MPa"] = y
        hatalar["tahmin_MPa"] = tahminler[ad]
        hatalar["mutlak_hata_MPa"] = np.abs(y - hatalar["tahmin_MPa"])
        hatalar["mutlak_yuzde_hata"] = 100 * hatalar["mutlak_hata_MPa"] / y
        hatalar.sort_values("mutlak_hata_MPa", ascending=False).to_csv(
            CIKTI / dosya, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    sinirlar = [min(y.min(), tahminler[list(modeller)].min().min()),
                max(y.max(), tahminler[list(modeller)].max().max())]
    for ax, ad in zip(axes.flat, modeller):
        ax.scatter(y, tahminler[ad], s=18, alpha=0.65, color="#2878a5")
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
    print("Dosyalar:", CIKTI)
    print("Sonuçlar geliştirme içindir; nihai test başarısı değildir.")


if __name__ == "__main__":
    main()
