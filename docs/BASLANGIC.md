# İlk adım: veriyi tanı

Hedef: veri setindeki yükleme için plakanın tamamındaki maksimum von Mises
gerilmesini MPa cinsinden tahmin etmek. Bu bir çatlak veya hasar tahmini değildir.

Girdiler: `dw`, `exw`, `eyw`, `lam`.
Hedef: `max_von_mises_MPa = 100 * kt_global`.
`kt` delik kenarının maksimumudur; bizim hedefimiz `kt_global` üzerinden hesaplanır.

Önce kök dizindeki `README.md` kurulumunu tamamla. Aşağıdaki `python`
komutları proje sanal ortamında çalıştırılmalıdır. Windows’ta `python` yerine
`.venv\Scripts\python.exe`, macOS/Linux’ta `.venv/bin/python` kullanılabilir.
Komutlar kurulmuş paket sayesinde farklı çalışma dizinlerinden de çalışır.

İlk komut:

```bash
python -m feng497 inspect
```

`.venv`, bu projeye ait Python ve kütüphaneleri tutar.
Paketi yeniden kurmak gerekirse:

```bash
python -m pip install -e .
```

Kodun anlamı:

- `pd.read_csv`: CSV dosyasını bir tablo olarak okur.
- `veri["split"] == "train"`: eğitim satırlarını seçen koşuldur.
- `.loc[...].copy()`: bu satırlardan ayrı bir tablo oluşturur.
- `100.0 * egitim["kt_global"]`: normalize sonucu MPa'ya çevirir.
- `X`: modelin kullanacağı dört girdi sütunudur.
- `y`: modelin tahmin etmeyi öğreneceği sonuçtur.
- `.head()`: ilk beş satırı gösterir.
- `.isna().sum()`: her sütundaki eksik değerleri sayar.

Bu dosya yalnızca veriyi okur ve eğitim grubunu inceler; model eğitmez.
Aktif veri sürümü v2'dir: 425 eğitim ve 110 test senaryosu bulunur.
Hazır test gruplarını koruyoruz. Model karşılaştırmaları yalnızca 425 eğitim
örneği içinde çapraz doğrulama ile yapılır. Son değerlendirmede 50 `test_interp`,
30 `test_mid` ve 30 `test_big` örneği ayrı raporlanır. V1'deki `test_extrap`,
çaplar artık eğitim aralığında olduğundan v2'de `test_mid` adını almıştır.

Veri sağlayıcısının `data/raw/plate_hole_dataset_v2/README.txt` dosyasındaki Kt başarıları delik kenarı hedefi içindir; kendi global maksimum
hedefimizin başarısı olarak kullanılamaz. Fiziksel geçerlilik pakette tanımlanan
2D çelik, doğrusal elastik model ve yükleme kapsamıyla sınırlıdır.

## İkinci adım: grafik çiz

```bash
python -m feng497 plot
```

Grafikler `reports/02_grafikler/` klasörüne kaydedilir.

- Histogramda yatay eksen gerilme aralıkları, dikey eksen simülasyon sayısıdır.
- Saçılım grafiklerinde yatay eksen bir girdi, dikey eksen hedef gerilmedir.
  Her nokta ayrı bir eğitim simülasyonudur.
- `ax.hist(...)` histogram çizer; `ax.scatter(x, y)` noktaları çizer.
- `plt.subplots(2, 2)` dört grafik alanı oluşturur.
- `fig.savefig(...)` grafiği PNG dosyası olarak kaydeder.

Diğer parametreler de örnekler arasında değiştiği için noktalar tek bir çizgi
üzerinde olmak zorunda değildir. Bunlar bir değişkenin diğerleri sabitken etkisini
ölçen kontrollü deneyler değildir. Pozitif ve negatif delik kaymalarında benzer
gerilmeler görülmesi mümkündür; doğrusal korelasyon tek başına özellik seçimi
için yeterli değildir. Bu aşamada dört girdiyi de koruyoruz.

Bu derste de test örnekleri çizilmez ve model eğitilmez.

## Üçüncü adım: ilk model ve çapraz doğrulama

```bash
python -m feng497 baseline
```

`KFold(n_splits=5, shuffle=True, random_state=42)` eğitimdeki 425 örneği
beş parçaya ayırır. Her turda dört parça (340 örnek) öğrenme, kalan parça
(85 örnek) doğrulama içindir. Bu işlem beş kez tekrarlanır.

`cross_val_predict(model, X, y, cv=cv)` bu beş turu yürütür. Her örneğin
tahmini, o örneği öğrenirken görmemiş bir modelden gelir. Bu, modelin eğitim
örneklerini ezberleyip ezberlemediğini değerlendirmek için gereklidir.

Ortalama referansı her turdaki 340 öğrenme örneğinin hedef ortalamasını
tahmin eder. Linear Regression ise dört girdiye katsayılar öğrenir:

```text
tahmin = sabit + a*dw + b*exw + c*eyw + d*lam
```

- MAE [MPa]: mutlak hataların ortalaması; düşük daha iyi.
- RMSE [MPa]: büyük hatalara daha fazla ağırlık veren ölçü; düşük daha iyi.
- R²: 1 mükemmel uyum, 0 gerçek hedeflerin ortalamasını tahmin etme düzeyi;
  negatif de olabilir. Referans modelin değeri, kat ortalamaları değiştiği için
  tam 0 olmak zorunda değildir.
- Ortalama mutlak yüzde hata: her örneğin mutlak hatası kendi gerçek
  değerine bölünür, yüzdeye çevrilir ve ortalanır.

CSV sonuçları ve gerçek–tahmin grafiği `reports/03_ilk_model/` içindedir.
Grafikte kesikli çizgi mükemmel tahmindir; noktaların ona yakın olması istenir.
Buradaki skorlar geliştirme amaçlı çapraz doğrulama sonuçlarıdır. Son test
başarısı değildir. Modeller aynı bölmelerle karşılaştırılır. Bu derste nihai
model kaydedilmez; seçilen model 425 örneğin tamamıyla eğitilir.

## Dördüncü adım: Random Forest

```bash
python -m feng497 random-forest
```

Yeni bir kütüphane gerekmez. Kod üçüncü dersteki aynı girdileri, aynı MPa
hedefini ve aynı beş doğrulama bölmesini kullanır. Linear Regression ve
ortalama referansı da yeniden çalıştırılır; sonuçları önceki dersle aynı olmalıdır.

```python
RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
```

- `n_estimators=300`: ormandaki karar ağacı sayısıdır; veri satırı sayısı değildir.
- `random_state=42`: aynı ortamda tekrar çalıştırmada rastgele seçimi sabitler.
- `n_jobs=1`: hesaplamayı tek işlemci işiyle yürütür.

Her ağaç girdi değerlerini eşiklerle bölerek öğrenir; örneğin `dw` belirli
bir değerden küçük mü diye sorabilir. Bu örnek eşikler bizim yazdığımız fizik
kuralları değildir; eğitimden öğrenilir. Orman ağaçların tahminlerini ortalar.
Beş turda beş ayrı orman eğitilir. Her orman yalnızca kendi 340 öğrenme
örneğini görür. Şimdilik ayar araması veya hedefe log dönüşümü yapılmaz.

`reports/04_random_forest/` içinde metrik tablosu, doğrulama tahminleri,
gerçek–tahmin grafiği ve en büyük mutlak hatalara göre sıralanmış RF kayıtları
bulunur. Yalnızca ortalama hataya bakma; yüksek gerilmelerdeki hataları da incele.
Bu başlangıç modeli, README'deki ayarlanmış RF deneyiyle aynı deney değildir.
Hedefimiz de delik kenarı maksimumu yerine tüm plakanın maksimumudur.

Random Forest'ın eğitim hedeflerinin dışına taşan tahmin yapmasını beklemiyoruz.
Eğitimden daha yüksek gerilmeli koşullarda bu önemli bir sınırlamadır. Modelin
bir doğrulama karşılaştırmasında iyi çıkması fiziksel güvenilirliği veya farklı
malzeme ve yüklemelere geçerliliği kanıtlamaz.

## Beşinci adım: Gaussian Process Regression (GPR)

```bash
python -m feng497 gpr
```

Bu ders iki GPR yaklaşımını, önceki Linear Regression ve Random Forest ile
aynı dört girdi ve aynı beş doğrulama bölmesinde karşılaştırır. Hazır test
gruplarına dokunulmaz. Yeni kütüphane kurulması gerekmez.

GPR, çekirdek (kernel) adı verilen bir benzerlik fonksiyonunu kullanarak
girdiler arasındaki ilişkiden tahmin üretir. Burada başlangıç seçimi, paketteki
örnek kodda kullanılan Matérn çekirdeğidir. Çekirdeğin sayısal parametreleri
her turun öğrenme grubunda optimize edilir; doğrulama hedefleri bu işleme katılmaz.

`make_pipeline(StandardScaler(), GaussianProcessRegressor(...))` ölçekleme
ve modeli bir arada tutar. Ölçekleme her girdiden öğrenme ortalamasını çıkarıp
öğrenme standart sapmasına böler. Doğrulama verisi aynı öğrenilmiş değerlerle
dönüştürülür. Böylece tüm 425 örneği önceden ölçekleyerek veri sızıntısı yaratmayız.

- `GPR`: doğrudan MPa hedefini öğrenir.
- `GPR (log hedef)`: MPa hedefinin doğal logaritmasını öğrenir; tahminde
  `exp` ile MPa'ya geri döner. Bu, göreli hataya odaklanan bir alternatif olarak
  denenir; daha iyi olacağı varsayılmaz.
- `TransformedTargetRegressor`: log/exp işlemlerini otomatik yapar. Dönen
  tahminler MPa'dır; tüm yöntemler aynı birimde değerlendirilir.
- `normalize_y=True`: GPR içinde o turun öğrenme hedefini standartlaştırır;
  log dönüşümünden ayrı bir işlemdir.

Log uzayındaki tahminin üstelini almak orijinal uzayda aritmetik beklenen
değeri hesaplamakla aynı değildir. Bu derste geri dönüşümü yapılmış nokta
tahminlerini değerlendiriyoruz; belirsizlik aralığı veya güvenlik garantisi
sunmuyoruz.

Metrikler, tahminler, iki GPR'nin hata tabloları ve grafik
`reports/05_gpr/` klasöründedir. Optimizasyon uyarıları varsa
`egitim_uyarilari.txt` içine yazılır; uyarı yoksa dosyada bu belirtilir.
Bu ders log dönüşümünün katkısını GPR modeli içinde karşılaştırır; model
ailesi ile hedef dönüşümünün etkilerini tek bir neden gibi yorumlamamalıyız.
Bu sonuçlarla model seçimi yapılabilir, fakat seçimin son performansı daha sonra
ayrılan test gruplarıyla ölçülecektir.

## Altıncı adım: son test ve modelin kaydedilmesi

```bash
python -m feng497 train
```

V1'de seçilmiş log hedefli GPR'nin ayarları korunur; v2'de 425 eğitim örneğinin
tamamıyla yeniden eğitilir. Sonra 50 küçük delik, 30 orta delik ve 30 büyük delik
testi ayrı değerlendirilir.
Test sonuçlarına göre bu kod model ayarı seçmez veya değiştirmez.

- `model.fit(X, y)` öğrenmeyi yapar. Scaler ve log hedef dönüşümü modelin içindedir.
- `model.predict(X_test)` öğrenilmiş modeli kullanarak MPa cinsinden tahmin üretir.
- `joblib.dump(model, yol)` öğrenilmiş modeli diske kaydeder.
- `joblib.load(yol)` modeli tekrar eğitim yapmadan kullanıma açar.

`models/` içinde model (`maksimum_von_mises_gpr.joblib`) ve
`model_bilgileri.json` bulunur. `reports/06_son_test/` ise test metriklerini,
her örneğin tahmin ve hatasını, grafiği ve eğitim uyarılarını içerir.
Kaydedilip yeniden yüklenen modelin 110 tahmini eğitimden hemen sonraki modelle
karşılaştırılarak kontrol edilir.

Bu model yalnızca tanımlanan fiziksel kapsam için değerlendirilmiştir. V2'deki
üç test grubunun girdileri eğitim aralıklarındadır; eğitim dışı koşullar için
ayrı bir ekstrapolasyon testi yoktur. Testlere baktıktan sonra bunlara göre
ayar değiştirirsek, yeni bağımsız performans iddiası için yeni test verisi gerekir.
Kaydedilen dosya ölçekleme ve log dönüşümünü de içerir; kullanırken girdilere
ayrıca ölçekleme veya sonuca ayrıca `exp`/100 çarpımı uygulanmaz.

## Yedinci adım: kendi girdilerinle tahmin al

```bash
python -m feng497 predict --cap 30 --ex 0 --ey 0 --lam 0
```

- `--cap 30`: delik çapı 30 mm (yarıçap değil).
- `--ex 0 --ey 0`: delik plakanın merkezinde, yani (150, 50) mm.
- `--lam 0`: yatay gerilme 100 MPa, dikey gerilme 0 MPa.

Bu değerler komutta değiştirilebilir. Ondalık ayıracı olarak nokta kullanılır.
Pozitif ex sağa, pozitif ey yukarı kaymadır; bunlar mutlak koordinatlar değildir.
Plaka boyutları, malzeme ve yatay nominal yük bu arayüzde değişmez.

Kod çapı ve kaymaları 100 mm plaka genişliğine böler. Böylece modelin
beklediği `dw, exw, eyw, lam` sırasıyla tek satırlık tablo oluşur.
`model.predict(X_yeni)` tahmini doğrudan MPa cinsinden verir; yeniden eğitim
yapılmaz. Burada gerçek ANSYS sonucu bulunmadığı için bu örneğin hatasını
hesaplayamayız. Gösterilen iki ondalık basamak fiziksel hassasiyet garantisi değildir.

Geometri plaka dışına taşıyorsa veya girdiler sonlu değilse kod durur. Herhangi
bir girdi eğitim min–max sınırları dışındaysa ya da dikey kenara kalan mesafe
10 mm'den azsa kapsam mesajı gösterilir. Tekil aralıkların içinde olmak da aynı
parametre kombinasyonunun daha önce görüldüğü veya doğru tahmin edileceği
anlamına gelmez. V1'deki ekstrapolasyon testi de v2'de artık eğitimde görülen
çap aralığındadır; eğitim dışı durumlar ayrıca doğrulanmış değildir.

V1 modeli `models/archive/v1/`, önceki grafik ve sonuçlar `reports/archive/v1/`
altında saklanır. Aktif `models/` ve `reports/02_...`–`06_...` çıktıları v2 içindir.
