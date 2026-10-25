# V2 verisiyle yeniden eğitim — 25 Ekim 2026

Veri sağlayıcısının v2 paketi 535 senaryo içeriyor: v1'deki 380 senaryo ve
155 yeni büyük delik senaryosu. Eski senaryoların ortak sütunları birebir
karşılaştırıldı; sayısal değerler değişmemiş. İki sürüm birleştirilmedi;
aktif veri dosyası yalnızca v2'nin `cases.csv` dosyasıdır.

## Veri grupları

| Grup | Senaryo | Gerçek çap aralığı [mm] |
|---|---:|---:|
| Eğitim | 425 | 10.11–80.00 |
| `test_interp` | 50 | 10.89–59.03 |
| `test_mid` | 30 | 60.01–65.00 |
| `test_big` | 30 | 60.39–79.90 |

V1'deki `test_extrap`, v2'de `test_mid` adını aldı; aynı 30 senaryo testte
kalmaya devam ediyor. V1'in hiçbir test senaryosu eğitime taşınmadı. Yeni
155 senaryonun 125'i eğitim, 30'u test grubunda. Üç test grubunun girdileri
artık eğitim aralıklarında; v2'de ayrı bir ekstrapolasyon testi yok.

## Eğitim ve değerlendirme

Girdiler `dw, exw, eyw, lam`; hedef `100 * kt_global` [MPa]. Hedef, delik
kenarının değil bütün plakanın maksimumudur. Nominal yatay gerilme 100 MPa,
plaka 300 × 100 × 2 mm ve malzeme doğrusal elastik çelik olarak korunuyor.

V1'de seçilen StandardScaler + Matérn GPR + log/exp hedef dönüşümü korundu.
Çekirdek başlangıç ayarları, üç optimizasyon yeniden başlatması ve rastgelelik
tohumları değiştirilmedi. Öğrenilen çekirdek parametreleri yeni eğitim verisinde
yeniden hesaplandı. Test gruplarına göre ayar veya özellik seçilmedi.

425 eğitim senaryosunda aynı 5 katlı çapraz doğrulama yeniden çalıştırıldı.
Her turda 340 örnek öğrenme, 85 örnek doğrulama için kullanıldı.
CV ortalama mutlak yüzde hataları:

| Model | CV ortalama mutlak yüzde hata |
|---|---:|
| Linear Regression | %23.28 |
| Random Forest | %7.17 |
| GPR | %1.31 |
| GPR (log hedef) | %1.10 |

Sabitlenen log hedefli GPR ardından 425 eğitim senaryosunun tamamıyla eğitildi.
Kaydedilip tekrar yüklenen modelin 110 test tahmini, kayıt öncesi tahminlerle
eşleşti (`rtol=atol=1e-12`). Son eğitimde optimizasyon uyarısı oluşmadı.
CV'de 6 çekirdek üst sınırı uyarısı kaydedildi; ayrıntılar
`reports/05_gpr/egitim_uyarilari.txt` dosyasındadır. Bu uyarılar üzerine testlere
göre ayar değişikliği yapılmadı.

## Aynı 110 senaryoda eski ve yeni model

| Test grubu | V1 ortalama hata | V2 ortalama hata | V2 en yüksek hata |
|---|---:|---:|---:|
| Küçük delikler (`test_interp`) | %0.45 | %0.64 | %3.76 |
| Orta delikler (`test_mid`) | %1.98 | %1.01 | %5.51 |
| Büyük delikler (`test_big`) | %11.23 | %1.49 | %7.50 |

Bunlar her senaryonun ANSYS maksimumuna göre mutlak yüzde hatalardır.
Orta ve büyük deliklerde iyileşme var; küçük deliklerde ortalama yüzde hata
biraz yükseldi. Bu sonuçlar yeni senaryolar için hata garantisi değildir.
V1 modeli büyük deliklerde eğitim çap aralığının dışında kullanılıyor;
v2 modeli bu çaplarda eğitim örneklerine sahip.

Tam metrikler `reports/06_son_test/v1_v2_karsilastirma.csv` içindedir.
Karşılaştırma, arşivlenmiş v1 modeline de aynı 110 test girdisi verilerek
hesaplandı. V1'de daha önce incelenmiş 80 test senaryosu geçmişle karşılaştırma
niteliğindedir; bu çalışma boyunca testler yeniden ayar yapmak için kullanılmadı.

## Dosyalar ve tekrar çalıştırma

- `data/raw/plate_hole_dataset_v2/`: cases.csv, sağlayıcı README'si ve ANSYS doğrulama tabloları.
- `data/raw/plate_hole_dataset_v1/`: değişmeden korunan eski veri.
- `models/archive/v1/`: değişmeden korunan eski model ve bilgileri.
- `reports/archive/v1/`: eski CV, grafikler ve test sonuçları.
- `models/`: aktif v2 modeli ve veri/model hash'lerini içeren metadata.
- `reports/05_gpr/`: 425 eğitim senaryosunun çapraz doğrulama tahminleri.
- `reports/06_son_test/`: 110 test tahmini, metrikler ve üç panelli karşılaştırma grafiği.

Büyük Parquet alan/mesh dosyaları ve delik profilleri bu scalar maksimum
gerilme modeli için gerekmediğinden depoya kopyalanmadı. Tam kaynak paket
İndirilenler'deki `plate_hole_dataset_v2` klasöründe duruyor.

Sanal ortam Python'u ile sırayla çalıştırılan komutlar:

```bash
python -m feng497 inspect
python -m feng497 plot
python -m feng497 baseline
python -m feng497 random-forest
python -m feng497 gpr
python -m feng497 train
python -m unittest discover -s tests -v
```

`train`, aktif model ve test raporlarını yeniden yazar; arşivdeki v1 dosyalarını
değiştirmez. `v1_v2_karsilastirma.csv` bu geçişteki model çiftine ait karşılaştırma
kaydıdır; `train` bu karşılaştırma dosyasını otomatik yenilemez.

Örnek merkezî 30 mm delik, `lam=0` için yeni tahmin 337.76 MPa'dır.
Yedi otomatik test; 110 tahminin eşleşmesini, modelin yalnızca eğitim satırlarıyla
öğrenmesini, eski verinin korunmasını, veri/model hash'lerini, geçersiz verinin
reddedilmesini ve terminal kullanımını denetler. CI bu testleri yeniden eğitim
yapmadan çalıştırır.
