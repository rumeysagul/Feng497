# FENG497: maksimum von Mises gerilmesi tahmini

Delikli bir çelik plakanın geometri ve yükleme parametrelerinden, plakanın
tamamındaki en yüksek von Mises gerilmesini MPa cinsinden tahmin eder.
Bu ilk aşama statik gerilme tahminidir. Çatlak boyutu veya ilerlemesini içermez.

## Proje yapısı

```text
Feng497/
├── src/feng497/          # Python paketi ve terminal komutları
│   ├── __main__.py      # Komut seçimi
│   ├── paths.py         # Ortak dosya yolları ve girdi sütunları
│   ├── modeling.py      # Ortak GPR oluşturma fonksiyonu
│   ├── inspect_data.py  # Veri inceleme
│   ├── plot_data.py     # Eğitim verisinin grafikleri
│   ├── baseline.py      # Ortalama ve doğrusal regresyon
│   ├── random_forest.py # Random Forest karşılaştırması
│   ├── gpr.py           # GPR karşılaştırması
│   ├── train.py         # Son eğitim, test ve model kaydı
│   └── predict.py       # Kaydedilmiş modelle tahmin
├── data/raw/plate_hole_dataset_v1/
│   ├── README.md        # Veri sağlayıcısının orijinal açıklaması
│   └── data/cases.csv   # 380 simülasyonun sayısal özeti
├── models/              # Eğitilmiş model ve kapsam/sürüm bilgileri
├── reports/             # Grafikler, metrikler ve tahmin tabloları
├── docs/                # Adım adım kullanım rehberi
├── tests/               # Paket, dosya yolları ve tahmin regresyon kontrolleri
├── pyproject.toml       # Paket kurulumu ve terminal giriş noktası
└── requirements.txt     # Sabitlenmiş bağımlılık sürümleri
```

## Kurulum

Python 3.9–3.12 gerekir. Windows örnekleri Python 3.11 kullanır.
Komutları deponun kök dizininde çalıştırın.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m feng497 --help
.\.venv\Scripts\python.exe -m feng497 predict --cap 30 --ex 0 --ey 0 --lam 0
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m feng497 --help
.venv/bin/python -m feng497 predict --cap 30 --ex 0 --ey 0 --lam 0
```

`pip install -e .`, `src/` altındaki paketi geliştirme modunda kurar ve
`requirements.txt` bağımlılıklarını da yükler. Kaynak kodundaki değişiklikler
yeniden kurulum gerektirmeden kullanılır. Ortam etkinleştirilmişse aşağıdaki
komutlarda doğrudan `python` veya `feng497` kullanılabilir.

VS Code'da **Python: Select Interpreter** komutuyla bu deponun `.venv`
ortamını seçin. Ortam dosyaları Git'e eklenmez.

## Komutlar

| Komut | İşlev | Çıktı konumu |
|---|---|---|
| `python -m feng497 inspect` | Eğitim verisini ve eksik değerleri inceleme | Terminal |
| `python -m feng497 plot` | Eğitim verisinin dağılım grafikleri | `reports/02_grafikler/` |
| `python -m feng497 baseline` | Ortalama ve doğrusal regresyon karşılaştırması | `reports/03_ilk_model/` |
| `python -m feng497 random-forest` | Random Forest karşılaştırması | `reports/04_random_forest/` |
| `python -m feng497 gpr` | Doğrudan ve log hedefli GPR karşılaştırması | `reports/05_gpr/` |
| `python -m feng497 train` | Sabit modelle son eğitim ve ayrılmış testler | `models/`, `reports/06_son_test/` |
| `python -m feng497 predict --cap 30 --ex 0 --ey 0 --lam 0` | Hazır modelle tahmin | Terminal |

`train` komutu mevcut modeli ve son test raporlarını yeniden yazar.
Yalnızca tahmin almak için yeniden eğitim gerekmez.
Python dosyalarını doğrudan çalıştırmak yerine `python -m feng497 ...`
komutlarını kullanın. Modül import etmek eğitim başlatmaz.

## Girdiler ve fiziksel kapsam

- `--cap`: delik çapı, mm. Yarıçap değildir.
- `--ex`, `--ey`: plaka merkezinden kaymalar, mm. Sağ ve yukarı yön pozitiftir.
- `--lam`: dikey/yatay nominal gerilme oranı.

Plaka 300 × 100 × 2 mm, malzeme izotropik doğrusal elastik çeliktir.
Yatay nominal gerilme 100 MPa, dikey nominal gerilme `lam * 100 MPa` olarak
tanımlıdır. Komut bu kapsam dışındaki malzeme veya plaka boyutlarını değiştirmez.

Model girdileri `dw, exw, eyw, lam`, hedefi `100 * kt_global` [MPa] değeridir.
`kt_global` bütün plakanın maksimumunu, `kt` yalnızca delik kenarının
maksimumunu ifade eder. Aynı hedefler değildir.

Model ölçekleme ve log/exp dönüşümünü içerir. `predict` çıktısı MPa'dır,
sonuca tekrar `exp` uygulanmaz veya 100 ile çarpılmaz.
Örnek merkezî 30 mm delik için beklenen sonuç yaklaşık **338.18 MPa**'dır.

## Veri ve model dosyaları

Veri yolu `data/raw/plate_hole_dataset_v1/data/cases.csv` olarak tanımlıdır.
Sağlayıcının README'si tam veri paketini anlatır. Büyük alan/mesh dosyaları,
profiller ve ANSYS makroları bu küçük depoya dahil değildir.

- `models/maksimum_von_mises_gpr.joblib`: kaydedilmiş log hedefli GPR.
- `models/model_bilgileri.json`: girdi sırası, birimler, eğitim kapsamı ve sürümler.
- `reports/06_son_test/test_sonuclari.csv`: son test metrikleri.
- `reports/06_son_test/test_tahminleri.csv`: her test örneğinin sonucu ve hatası.

Dosya yolları `src/feng497/paths.py` içinde tanımlıdır. Geliştirme kurulumu,
komutu hangi klasörden çalıştırdığınıza bağlı olmadan bu depoyu bulur.
Başka bir veri/model deposunu kullanmak için `FENG497_ROOT` ortam değişkenini,
aynı `data/`, `models/`, `reports/` düzenini içeren kök klasöre ayarlayabilirsiniz.
Normal paket kurulumu kullanırsanız bu değişkeni veri deposuna yönlendirin.

## Değerlendirme

300 eğitim örneği içinde beş katlı çapraz doğrulama ile model karşılaştırılır.
50 interpolasyon ve 30 ekstrapolasyon örneği son değerlendirme içindir.
Test sonuçlarına göre yeni ayarlar seçilirse yeni bağımsız test verisi gerekir.

Depodaki mevcut son test sonuçları:

| Grup | Ortalama mutlak yüzde hata | Maksimum mutlak yüzde hata |
|---|---:|---:|
| İnterpolasyon | %0.45 | %2.81 |
| Ekstrapolasyon | %1.98 | %12.47 |

Bu değerler yeni tasarımlar için hata garantisi değildir. Veri sağlayıcısının
README'sindeki delik kenarı `Kt` başarıları bu global hedefle karıştırılmamalıdır.
Veri sağlayıcısı kamuya açık yayın öncesinde izin istemektedir.

## Kontroller ve rehber

### GitHub Actions

[CI workflow](.github/workflows/ci.yml), `main` ve `project-structure` branch'lerine
push yapıldığında ve `main` hedefli PR açıldığında veya güncellendiğinde çalışır.
Python 3.11 ile Ubuntu ve Windows üzerinde projeyi kurar, bağımlılık uyumunu,
terminal giriş noktasını ve aşağıdaki entegrasyon testlerini kontrol eder.

Model yeniden eğitilmez; kayıtlı modelin mevcut tahminleri koruyup korumadığı
denetlenir. Bu kontrol yazılımın çalışmasını denetler; yeni senaryolardaki fiziksel
doğruluğun veya ANSYS doğrulamasının yerine geçmez.

Sonuçlar GitHub'ın **Actions** sekmesinde ve PR'ın **Checks** bölümünde görünür.
CI başarısız olduğunda birleştirmenin engellenmesi için ayrıca repo yöneticisinin
bu kontrolleri `main` koruma kuralında zorunlu yapması gerekir.

### Yerel kontroller

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kontroller, kaydedilmiş modelin eski 80 test tahminini koruduğunu, veri hash'ini,
farklı çalışma dizininden tahmin alınabildiğini ve import sırasında eğitim
başlamadığını doğrular. Bu kontroller model seçimi veya yeniden eğitim yapmaz.

Adım adım açıklamalar [docs/BASLANGIC.md](docs/BASLANGIC.md) içindedir.
Eski paylaşım paketinin hash listesi, geçmiş kaydı olarak
`docs/original-package-checksums.json` altında saklanır. Oradaki eski yollar
ve kaynak kodu hash'leri bu yeni klasör düzeni için bir doğrulama listesi değildir.
