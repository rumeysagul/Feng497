# FENG497: çelik gerilmesi ve kompozit ilk hasar yükü tahmini

Delikli bir çelik plakanın geometri ve yükleme parametrelerinden, plakanın
tamamındaki en yüksek von Mises gerilmesini MPa cinsinden tahmin eder.
Bu ilk aşama statik gerilme tahminidir. Çatlak boyutu veya ilerlemesini içermez.

Yeni kompozit aşaması, aynı geometride T300/5208 karbon/epoksi tek yönlü tek
tabakanın Tsai-Wu ilk hasar yükünü tahmin eder. Fiber açısı beşinci fiziksel
girdidir. Kompozit komutları, veri seti ve model dosyaları çelikten ayrıdır.

## Kompozit verisiyle eğitim ve tahmin

Veri sağlayıcısının `UD_composite_dataset_v1` paketinden `cases.csv`, açıklama
ve küçük doğrulama tabloları `data/raw/UD_composite_dataset_v1/` içine alınmıştır.
808 senaryonun 648'i eğitim, 100'ü `test_interp`, 60'ı `test_0_90` grubundadır.
Son grupta fiber açısı tam 0° olan 30 ve tam 90° olan 30 senaryo vardır.

Windows PowerShell, proje kökünde:

```powershell
.\.venv\Scripts\python.exe -m feng497 inspect-composite
.\.venv\Scripts\python.exe -m feng497 train-composite
.\.venv\Scripts\python.exe -m feng497 predict-composite --cap 30 --ex 0 --ey 0 --lam 0 --fib 0
.\.venv\Scripts\python.exe -m feng497 predict-composite --cap 30 --ex 0 --ey 0 --lam 0 --fib 90
```

`--fib` +x ekseninden ölçülen fiber açısıdır; `--fiber-angle` da kullanılabilir.
`lam=0` için yatay çekme altında `fib=0` fiberlere paralel, `fib=90` fiberlere
dik yüklemeyi temsil eder. `lam` sıfırdan farklıysa ek dikey gerilme de vardır.

**Hedef `100 * tw_R_global` [MPa]**: sabit yük oranıyla tüm yükler birlikte
ölçeklenirken plakanın herhangi bir yerinde ilk hasarı başlatan nominal
`sigma_x`. `sigma_y = lam * sigma_x`. Veri paketindeki `fail_load_tw_mpa`
yalnızca delik kenarını kapsar; 14 senaryoda ilk hasar başka yerde başladığından
global hedef farklıdır. Sağlayıcının delik kenarı için verdiği referans hata
oranları bu yeni global hedefin sonuçlarıyla doğrudan karşılaştırılamaz.

Model yalnızca `D, ex, ey, lam, fib` girdilerinden 11 özellik hesaplar: normalize
geometri, `cos(2*fib)`, `sin(2*fib)` ve kenar mesafeleri. Gerilme, hasar sonucu,
mesh bilgisi veya test etiketleri girdi olarak kullanılmaz. Sabit log hedefli
GPR ve doğrusal regresyon referansı, yalnızca 648 eğitim örneğinde aynı beş CV
bölmesiyle incelenir. Son GPR tüm eğitim grubuyla öğrenir. İki ayrılmış test
grubu final değerlendirmede kullanılır; 0° ve 90° sonuçları ayrıca raporlanır.
Test sonuçlarına göre hiperparametre değiştirilmez.

- Model: `models/composite_v1/kompozit_ilk_hasar_gpr.joblib`
- Kapsam, kaynak/model hash'leri ve eğitim kimlikleri: `models/composite_v1/model_bilgileri.json`
- Eğitim içi CV: `reports/composite_v1/cv_sonuclari.csv`
- Ayrılmış testler: `reports/composite_v1/test_sonuclari.csv`
- Fiber yönleri ayrı: `reports/composite_v1/test_0_90_sonuclari.csv`
- Senaryo tahminleri: `reports/composite_v1/test_tahminleri.csv`

Kaydedilmiş kompozit modelin test sonuçları:

| Grup | Senaryo | Ortalama mutlak hata (MPa) | Ortalama mutlak yüzde hata | En yüksek yüzde hata |
|---|---:|---:|---:|---:|
| Ara açılar | 100 | 0.195 | %0.73 | %5.79 |
| 0° ve 90° birlikte | 60 | 0.600 | %1.72 | %9.63 |
| 0° | 30 | 1.062 | %2.14 | %9.63 |
| 90° | 30 | 0.137 | %1.30 | %9.15 |

Beş katlı eğitim içi doğrulamada GPR'nin ortalama mutlak yüzde hatası %1.03,
doğrusal regresyon referansının %29.04'tür. Son GPR'nin kaydı yeniden
yüklenerek 160 test tahmininin aynı kaldığı kontrol edildi. GPR'nin bazı
çekirdek parametreleri optimizasyon sınırlarına yaklaştı; uyarılar
`reports/composite_v1/egitim_uyarilari.txt` içinde saklanır.

Merkezde 30 mm delik ve yalnızca yatay çekme (`lam=0`) için örnek ML tahmini,
fiber açısı 0°'de **82.36 MPa**, 90°'de **14.17 MPa**. Bu iki örnek tahmin
ayrı ANSYS çözümü veya test hatası olarak sunulmaz.

`train-composite` bu kompozit çıktılarını yeniden yazar. Eski çelik model ve
raporları korunur. Tam alan Parquet ve delik profil dosyaları, bu tek değer
modeli için gerekmediğinden depoya kopyalanmamıştır; eğitim `cases.csv` ile
çalışır. Tahmin komutu sağlayıcının geometri, minimum 10 mm kenar mesafesi,
yük oranı ve 0–90° açı kapsamının dışındaki girdileri reddeder.

Bu model sağlayıcının sabit malzeme dayanımları ve Tsai-Wu etkileşim katsayısı
varsayımına bağlıdır. Doğrusal elastik FE sonucundan ilk tabaka hasarı tahmin
eder; nihai taşıma kapasitesi, fiziksel deney doğrulaması, çatlak boyutu,
ilerlemesi veya hasar konumu modeli değildir. Çelik verileri bu eğitime katılmaz.
Von Mises, kompozit hasar hedefi olarak kullanılmaz.

Aşağıdaki mevcut ders komutları ve çelik sonuçları önceki aşamaya aittir.

## Proje yapısı

```text
Feng497/
├── src/feng497/          # Python paketi ve terminal komutları
│   ├── __main__.py      # Komut seçimi
│   ├── paths.py         # Ortak dosya yolları ve girdi sütunları
│   ├── dataset.py       # Veri ve test grubu doğrulamaları
│   ├── modeling.py      # Ortak GPR oluşturma fonksiyonu
│   ├── inspect_data.py  # Veri inceleme
│   ├── plot_data.py     # Eğitim verisinin grafikleri
│   ├── baseline.py      # Ortalama ve doğrusal regresyon
│   ├── random_forest.py # Random Forest karşılaştırması
│   ├── gpr.py           # GPR karşılaştırması
│   ├── train.py         # Son eğitim, test ve model kaydı
│   └── predict.py       # Kaydedilmiş modelle tahmin
├── data/raw/plate_hole_dataset_v2/
│   ├── README.txt       # Veri sağlayıcısının orijinal açıklaması
│   ├── data/cases.csv   # 535 simülasyonun sayısal özeti
│   └── validation/     # Sağlayıcının ANSYS doğrulama tabloları
├── data/raw/plate_hole_dataset_v1/ # Önceki veri sürümü
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
Örnek merkezî 30 mm delik için v2 modelinin sonucu yaklaşık **337.76 MPa**'dır.

## Veri ve model dosyaları

Veri yolu `data/raw/plate_hole_dataset_v2/data/cases.csv` olarak tanımlıdır.
V2, önceki 380 senaryoyu ve 155 yeni büyük delik senaryosunu birlikte içerir;
v1 ve v2 dosyaları üst üste eklenmez. Eğitimde 425, testlerde 110 senaryo vardır.
Eğitimde delik çapı 10.11–80 mm aralığındadır. Sağlayıcının README'si tam veri
paketini anlatır. Büyük alan/mesh ve profil dosyaları bu küçük depoya dahil değildir;
maksimum gerilme modeli yalnızca `cases.csv` kullanır.

- `models/maksimum_von_mises_gpr.joblib`: kaydedilmiş log hedefli GPR.
- `models/model_bilgileri.json`: girdi sırası, birimler, eğitim kapsamı ve sürümler.
- `reports/06_son_test/test_sonuclari.csv`: son test metrikleri.
- `reports/06_son_test/test_tahminleri.csv`: her test örneğinin sonucu ve hatası.
- `models/archive/v1/`: önceki 300 senaryoyla eğitilmiş model ve metadata.
- `reports/archive/v1/`: v1 grafiklerinin, CV ve test sonuçlarının değişmeden kopyası.
- `reports/06_son_test/v1_v2_karsilastirma.csv`: aynı 110 test senaryosunda iki modelin karşılaştırması.

Dosya yolları `src/feng497/paths.py` içinde tanımlıdır. Geliştirme kurulumu,
komutu hangi klasörden çalıştırdığınıza bağlı olmadan bu depoyu bulur.
Başka bir veri/model deposunu kullanmak için `FENG497_ROOT` ortam değişkenini,
aynı `data/`, `models/`, `reports/` düzenini içeren kök klasöre ayarlayabilirsiniz.
Normal paket kurulumu kullanırsanız bu değişkeni veri deposuna yönlendirin.

## Değerlendirme

425 eğitim örneği içinde beş katlı çapraz doğrulama ile model karşılaştırılır.
50 `test_interp`, 30 `test_mid` ve 30 `test_big` örneği ayrı ayrı değerlendirilir.
Eski `test_extrap` grubu v2'de `test_mid` adını almıştır. Çaplar artık eğitim
aralığında olduğundan bu grup ekstrapolasyon testi olarak adlandırılmaz.
GPR yapısı, dört girdi ve `100 * kt_global` hedefi önceki sürümle aynıdır.
Test sonuçlarına göre yeni ayarlar seçilirse yeni bağımsız test verisi gerekir.

Depodaki mevcut son test sonuçları:

| Grup | Ortalama mutlak yüzde hata | Maksimum mutlak yüzde hata |
|---|---:|---:|
| `test_interp` — 50 küçük delik | %0.64 | %3.76 |
| `test_mid` — 30 orta delik | %1.01 | %5.51 |
| `test_big` — 30 büyük delik | %1.49 | %7.50 |

Aynı test senaryolarında v1 modelinin ortalama yüzde hataları sırasıyla %0.45,
%1.98 ve %11.23'tür. V2, orta ve büyük deliklerde daha iyi sonuç verdi; küçük
deliklerde ortalama yüzde hata hafifçe yükseldi. Ek veri her grupta otomatik
iyileşme sağlamaz. Karşılaştırma sonrası testlere göre ayar değiştirilmedi.

V2 geçişinin kapsamı ve çalıştırılan adımlar [docs/V2_GECIS.md](docs/V2_GECIS.md)
içinde kayıtlıdır.

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

Kontroller, çelik modelinin 110 ve kompozit modelinin 160 kayıtlı test
tahminini, veri/model hash'lerini, eski çelik senaryolarının korunduğunu ve
her iki modelin yalnızca eğitim satırlarıyla öğrendiğini doğrular. Farklı
çalışma dizininden tahmin, kompozit açı ve geometri kontrolleri ve import
sırasında eğitim başlamaması da denetlenir. Bu kontroller model seçimi veya
yeniden eğitim yapmaz.

Adım adım açıklamalar [docs/BASLANGIC.md](docs/BASLANGIC.md) içindedir.
Eski paylaşım paketinin hash listesi, geçmiş kaydı olarak
`docs/original-package-checksums.json` altında saklanır. Oradaki eski yollar
ve kaynak kodu hash'leri bu yeni klasör düzeni için bir doğrulama listesi değildir.
