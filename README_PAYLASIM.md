# FENG497 ML proje paketi

ZIP'i tamamen çıkarın ve komutları çıkan klasörün içinde çalıştırın.
Öğrenme raporları bu pakette bulunmaz; ayrı paylaşılır.

## Proje
Dört girdi: dw, exw, eyw, lam. Hedef: 100*kt_global [MPa], plakanın
TAMAMINDAKİ maksimum von Mises gerilmesi. Eğitimli log hedefli GPR dahildir.
300 eğitim örneği, 50 interpolasyon ve 30 ekstrapolasyon testi kullanılmıştır.
Ortalama mutlak yüzde test hataları sırasıyla %0.45 ve %1.98; maksimum
hatalar %2.81 ve %12.47. Bunlar yeni tasarımlara hata garantisi değildir.

Fiziksel kapsam: 300x100x2 mm izotropik doğrusal elastik çelik plaka,
sigma_x=100 MPa, sigma_y=lam*100 MPa. Çatlak veya hasar tahmini değildir.

## Kurulum
Test edilen ortam Python 3.9.13 ve scikit-learn 1.6.1'dir. Bağımlılıklar
Python 3.9-3.12 ile uyumludur; diğer işletim sistemlerinde ayrıca denenmedi.
Windows'ta aşağıdaki komut Python 3.11'in kurulu olmasını gerektirir.

Windows:
    py -3.11 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe 07_tahmin_yap.py --cap 30 --ex 0 --ey 0 --lam 0

macOS veya Linux (uyumlu Python sürümüyle):
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python 07_tahmin_yap.py --cap 30 --ex 0 --ey 0 --lam 0

Beklenen yaklaşık sonuç: 338.18 MPa.
--cap delik ÇAPI [mm]; --ex ve --ey plaka merkezinden kaymalar [mm];
--lam dikey/yatay nominal gerilme oranıdır. Ondalık için nokta kullanın.

## Dosyalar
01-05: veri inceleme ve çapraz doğrulama/model karşılaştırması.
06: 300 eğitim örneğiyle yeniden eğitim, son test ve modeli kaydetme.
07: hazır modelden yeni tahmin. Sadece tahmin için yeniden eğitim gerekmez.
06 çalıştırılırsa mevcut model ve test çıktılarını yeniden yazar.

outputs/06_son_test/maksimum_von_mises_gpr.joblib: eğitimli model.
outputs/06_son_test/model_bilgileri.json: kapsam, girdi sırası ve sürümler.
outputs/06_son_test/test_sonuclari.csv: test metrikleri.
outputs/06_son_test/test_tahminleri.csv: her test örneğinin tahmini ve hatası.
outputs/05_gpr/cv_sonuclari.csv: model karşılaştırması.
BASLANGIC.md: kodları çalıştırmak için kısa proje rehberi.

Model ölçekleme ve log/exp dönüşümünü içerir; predict çıktısı MPa'dır.
Çıktıya tekrar exp uygulamayın veya 100 ile çarpmayın. Yalnızca ekibin
ürettiği/güvendiği joblib dosyalarını yükleyin.

## Veri kapsamı ve değerlendirme
cases.csv ve orijinal veri README'si dahildir. README tam veri paketini
anlatır; büyük Parquet alanları, mesh, profiller, ANSYS makroları ve örnek
quickstart bu küçük pakette bulunmaz. 01-07 kodları bunları gerektirmez.
.venv gönderilmez; herkes kendi ortamını oluşturur.
Hazır testlere göre yeni ayar seçilirse bağımsız değerlendirme için yeni
test verisi gerekir. Veri sağlayıcısı kamuya açık yayın öncesi izin ister.
