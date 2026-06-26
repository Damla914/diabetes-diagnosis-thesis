Bitirme Tezi Adı: Makine Öğrenmesi ile Diyabet Teşhisinde Sınıf Dengesizliği ve Karar Eşiği Optimizasyonu  
Yazar: Damla Hilal Erden
Yıl: 2026

1. PROJE VE KLASÖR YAPISI
--------------------------------------------------------------------------------
Klasör yapısı şu şekildedir:

├── 1_Rapor         --> Tezin metni (PDF formatında)
├── 2_Kod           --> Modelleme ve analiz süreçlerine ait kaynak kodlar
     ├── main.py                 --> Tüm kodlamayı çalıştıran ve yöneten dosya
     ├── data_check.py           --> Ham veri yükleme, eksik/geçersiz sıfır değer ve outlier analizi
     ├── preprocess.py           --> Veri temizleme ve yapılandırma adımları
     ├── scaling.py              --> StandardScaler tabanlı özellik ölçekleme modülü
     ├── train_test_split.py     --> Stratified veri bölünmesi ve görselleştirmesi
     ├── smote.py                --> SMOTE ile sentetik azınlık aşırı örnekleme adımı
     ├── nested_cv.py            --> İç İçe Çapraz Doğrulama 
     ├── logistic_regression.py  --> Lojistik Regresyon modelleri
     ├── knn.py                  --> KNN
     ├── decision_tree.py        --> Karar Ağaçları
     ├── random_forest.py        --> Rastgele Orman algoritması modülü
     ├── stacking.py             --> Stacking Ensemble modeli mimarisi
     ├── compare_models.py       --> Tüm modellerin metrik tablosu ve barplot karşılaştırması
     ├── comparative_tuning.py   --> Klinik Eşik Optimizasyonu ve FN/FP maliyet analizi grafikleri
     ├── calibration.py          --> Brier Skoru hesaplama ve Kalibrasyon Eğrisi güvenilirlik analizi
     └── shap_analysis.py        --> Dengeli LR modeli üzerinde SHAP ile XAI (Açıklanabilir Yapay Zeka)
├── 3_Veri          --> "Pima Indians Diabetes" veri setinin ham ve işlenmiş hali
└── 4_Grafikler     
     ├── Tezde_Kullanilanlar  --> Tez metnine dahil edilen grafikler
     └── Diger_Analizler      --> Arka planda çalıştırılan, veri analizine ait ek grafikler


2. KODLARIN ÇALIŞTIRILMASI
--------------------------------------------------------------------------------
Tüm mimari `main.py` üzerinden arka arkaya tetiklenecek şekilde kurgulanmıştır.
Terminal veya komut satırı üzerinden proje ana dizinine gelerek aşağıdaki 
komutla tüm kodlama çalıştırılır:
    
    python main.py

Not: `main.py` içinde `matplotlib.use('Agg')` ayarı yapılmıştır. Grafiklerin 
bellekte şişme yapmaması ve toplu analizde tıkanma yaşanmaması için maksimum 
açık figür sınırı 100 olarak limitlenmiştir. Kodun ürettiği tüm grafikler 
'Grafikler' klasöründe saklanmıştır. Üretilen performans tabloları ve 
metrik skorları doğrudan terminal ekranına çıktı olarak basılacaktır.


3. ÖNE ÇIKANLAR VE ANALİZLER
--------------------------------------------------------------------------------
* Veri Sızıntısı (Data Leakage) Koruması: `nested_cv.py` modülü içerisinde, 
  StandardScaler ve SMOTE adımları `imblearn.pipeline.Pipeline` yapısına entegre 
  edilerek İç Çapraz Doğrulama (Inner CV: 3-Fold GridSearch) ve Dış Çapraz Doğrulama 
  (Outer CV: 5-Fold) döngülerine sokulmuştur. Böylece test setinin eğitim 
  aşamasına sızması matematiksel olarak engellenmiştir.
* Klinik Eşik Optimizasyonu: `comparative_tuning.py` modülü, tip-2 diyabet gibi 
  hayati risk taşıyan süreçlerde standart 0.50 eşik değerinin ötesine geçerek 
  F1-skorunu maksimize eden optimal karar sınırlarını hesaplar ve Yanlış Negatif (FN) 
  vs. Yanlış Pozitif (FP) dağılımlarını eşik değişimine göre grafikleştirir.
* Tahmin Güvenilirliği (Kalibrasyon): `calibration.py` modülü, modellerin sadece 
  etiket tahmin başarısını değil, ürettikleri olasılıkların gerçek dünya 
  oranlarıyla ne kadar uyuştuğunu Brier Skoru kaybı ve Kalibrasyon Eğrileri 
  üzerinden histrogram destekli olarak analiz eder.

