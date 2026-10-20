"""Altıncı ders: sabitlenen modeli eğit, ayrı testlerde değerlendir ve kaydet."""

from pathlib import Path
import hashlib
import json
import platform
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "outputs" / "06_son_test"
CIKTI.mkdir(parents=True, exist_ok=True)
VERI_YOLU = KLASOR / "plate_hole_dataset_v1/data/cases.csv"
veri = pd.read_csv(VERI_YOLU)
girdiler = ["dw", "exw", "eyw", "lam"]
egitim = veri.loc[veri["split"] == "train"].copy()

# Beşinci derste seçilen ayarlar; test sonuçlarına göre değiştirilmiyor.
kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(4), nu=2.5)
          + WhiteKernel(1e-6, (1e-10, 1e-2)))
model = TransformedTargetRegressor(
    regressor=make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                 n_restarts_optimizer=3, random_state=0)),
    func=np.log, inverse_func=np.exp,
)

print("Sabitlenen log hedefli GPR, 300 eğitim örneğiyle eğitiliyor...", flush=True)
with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as kayit:
    warnings.simplefilter("always", ConvergenceWarning)
    model.fit(egitim[girdiler], 100.0 * egitim["kt_global"])
uyarilar = [f"{u.category.__name__}: {u.message}" for u in kayit]
(CIKTI / "egitim_uyarilari.txt").write_text(
    "\n".join(uyarilar) if uyarilar else "Eğitim uyarısı yok.\n", encoding="utf-8")

# Ölçekleyici, GPR ve hedef dönüşümü tek dosyada saklanır.
MODEL_YOLU = CIKTI / "maksimum_von_mises_gpr.joblib"
joblib.dump(model, MODEL_YOLU)
yuklenen_model = joblib.load(MODEL_YOLU)

sonuclar = []
tum_tahminler = []
for grup in ["test_interp", "test_extrap"]:
    test = veri.loc[veri["split"] == grup].copy()
    gercek = 100.0 * test["kt_global"]
    with threadpool_limits(limits=1):
        tahmin = model.predict(test[girdiler])
        yeniden_tahmin = yuklenen_model.predict(test[girdiler])
    np.testing.assert_allclose(tahmin, yeniden_tahmin, rtol=1e-12, atol=1e-12)
    if not np.isfinite(tahmin).all() or not (tahmin > 0).all():
        raise ValueError("Geçersiz tahmin bulundu.")
    hata = np.abs(gercek - tahmin)
    yuzde = 100.0 * hata / gercek
    sonuclar.append({
        "split": grup, "n": len(test),
        "MAE_MPa": mean_absolute_error(gercek, tahmin),
        "RMSE_MPa": np.sqrt(mean_squared_error(gercek, tahmin)),
        "R2": r2_score(gercek, tahmin),
        "ortalama_mutlak_yuzde_hata": np.mean(yuzde),
        "maksimum_mutlak_yuzde_hata": np.max(yuzde),
    })
    tablo = test[["case", "split"] + girdiler].copy()
    tablo["gercek_MPa"] = gercek
    tablo["tahmin_MPa"] = tahmin
    tablo["mutlak_hata_MPa"] = hata
    tablo["mutlak_yuzde_hata"] = yuzde
    tum_tahminler.append(tablo)

rapor = pd.DataFrame(sonuclar)
tahminler = pd.concat(tum_tahminler, ignore_index=True)
rapor.to_csv(CIKTI / "test_sonuclari.csv", index=False)
tahminler.to_csv(CIKTI / "test_tahminleri.csv", index=False)

bilgiler = {
    "model": "GPR, Matérn nu=2.5, log hedef",
    "secim": "300 eğitim örneğinde 5 katlı CV; random_state=42",
    "girdiler_sirayla": girdiler,
    "hedef": "100 * kt_global", "hedef_birimi": "MPa",
    "egitim_ornek_sayisi": len(egitim),
    "girdi_egitim_araliklari": {
        s: {"min": float(egitim[s].min()), "max": float(egitim[s].max())}
        for s in girdiler},
    "veri_sha256": hashlib.sha256(VERI_YOLU.read_bytes()).hexdigest(),
    "python": platform.python_version(), "sklearn": sklearn.__version__,
    "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
    "ogrenilen_kernel": str(model.regressor_.named_steps["gaussianprocessregressor"].kernel_),
    "yeniden_yukleme_kontrolu": "80 test tahmini eşleşti; rtol=atol=1e-12",
    "kapsam": "300x100x2 mm plaka, izotropik doğrusal elastik çelik, "
              "plane stress, sigma_x=100 MPa, sigma_y=lam*100 MPa",
    "not": "Tam alan veya çatlak tahmini değildir. Testler model seçimi için kullanılmadı. "
           "Bu sonuçlardan sonra test verisine göre ayar değiştirmek bağımsız test niteliğini bozar.",
}
(CIKTI / "model_bilgileri.json").write_text(
    json.dumps(bilgiler, ensure_ascii=False, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
sinirlar = [min(tahminler["gercek_MPa"].min(), tahminler["tahmin_MPa"].min()),
            max(tahminler["gercek_MPa"].max(), tahminler["tahmin_MPa"].max())]
for ax, grup, baslik in zip(axes, ["test_interp", "test_extrap"],
                           ["İnterpolasyon — 50 örnek", "Ekstrapolasyon — 30 örnek"]):
    tablo = tahminler.loc[tahminler["split"] == grup]
    ax.scatter(tablo["gercek_MPa"], tablo["tahmin_MPa"], s=30, alpha=0.8)
    ax.plot(sinirlar, sinirlar, "--", color="#555555")
    ax.set_title(baslik)
    ax.set_xlabel("Gerçek maksimum von Mises [MPa]")
    ax.set_ylabel("Test tahmini [MPa]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
fig.suptitle("Sabitlenen log hedefli GPR — ayrılmış test grupları")
fig.tight_layout()
fig.savefig(CIKTI / "test_gercek_ve_tahmin.png", dpi=160)
plt.close(fig)

print(rapor.round(3).to_string(index=False))
print("\nHer test grubunda yüzde hatası en yüksek örnek:")
for grup in ["test_interp", "test_extrap"]:
    print(tahminler.loc[tahminler["split"] == grup]
          .nlargest(1, "mutlak_yuzde_hata").round(3).to_string(index=False))
print("\nModel kaydedildi ve yeniden yüklenen modelin 80 tahmini doğrulandı.")
print("Eğitim uyarısı sayısı:", len(uyarilar))
print("Dosyalar:", CIKTI)
