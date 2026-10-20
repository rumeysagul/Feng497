"""İkinci ders: yalnızca eğitim verisini grafiklerle incele."""

from pathlib import Path
import pandas as pd
import matplotlib

# Grafikler pencere açmak yerine PNG dosyasına kaydedilir.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "outputs" / "02_grafikler"
CIKTI.mkdir(parents=True, exist_ok=True)

veri = pd.read_csv(KLASOR / "plate_hole_dataset_v1/data/cases.csv")
egitim = veri.loc[veri["split"] == "train"].copy()
egitim["max_von_mises_MPa"] = 100.0 * egitim["kt_global"]
hedef = "max_von_mises_MPa"

# Histogram: bir gerilme aralığına kaç simülasyon düştüğünü gösterir.
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(egitim[hedef], bins=20, color="#2878a5", edgecolor="white")
ax.set_xlabel("Plakadaki maksimum von Mises gerilmesi [MPa]")
ax.set_ylabel("Simülasyon sayısı")
ax.set_title("Eğitim verisindeki 300 simülasyonun hedef dağılımı")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(CIKTI / "01_hedef_dagilimi.png", dpi=160)
plt.close(fig)

# Saçılım grafiği: her nokta, ayrı bir simülasyonu temsil eder.
girdiler = {
    "dw": "Delik çapı / plaka genişliği (dw)",
    "exw": "Yatay delik kayması / plaka genişliği (exw)",
    "eyw": "Dikey delik kayması / plaka genişliği (eyw)",
    "lam": "Dikey / yatay uygulanan gerilme (lam)",
}
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
for ax, (sutun, aciklama) in zip(axes.flat, girdiler.items()):
    ax.scatter(egitim[sutun], egitim[hedef], s=22, alpha=0.65,
               color="#2878a5", edgecolors="none")
    ax.set_xlabel(aciklama)
    ax.set_ylabel("Maksimum von Mises [MPa]")
    ax.grid(alpha=0.2)
fig.suptitle("Girdiler ve maksimum gerilme — her nokta bir eğitim simülasyonu")
fig.tight_layout()
fig.savefig(CIKTI / "02_girdiler_ve_gerilme.png", dpi=160)
plt.close(fig)

print("Çizilen eğitim örneği:", len(egitim))
print("Grafiklerin kaydedildiği klasör:", CIKTI)
print("Not: Her grafikte diğer üç parametre de değişiyor.")
print("Bu grafikler tek bir parametrenin izole fiziksel etkisini göstermez.")
