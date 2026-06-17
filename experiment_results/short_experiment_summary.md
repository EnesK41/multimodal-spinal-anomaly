# Kisa Deney Ozeti

- Dataset temizligi ve split kontrolu yapildi.
  - Neden: Augmentasyonlarin validation'a sizmasini engellemek ve ayni hastanin tum versiyonlarini ayni splitte tutmak.
  - Sonuc: 42 base patient ile patient-level 5-fold CV kullanildi.

- Prompt metni modelden temizlendi.
  - Neden: Prompt icinde tani/bolge bilgisi olabilecegi icin shortcut riski vardi.
  - Sonuc: X-ray deneyleri sadece goruntu uzerinden de calisir hale geldi.

- Mask-derived binary label kullanildi.
  - Neden: CSV/prompt kaynakli olasi hatalari azaltmak ve label'i mask var/yok bilgisinden almak.
  - Sonuc: Anomaly/healthy sinifi daha tutarli hale geldi.

- Body-center ROI crop eklendi.
  - Neden: Modelin goruntunun ilgisiz kenar/bilgi kisimlarina odaklanmasini azaltmak.
  - Sonuc: En iyi X-ray-only ayar ROI 0.45 oldu.

- ImageNet-pretrained ResNet34 fine-tuning denendi.
  - Neden: Veri cok az oldugu icin sifirdan egitim zayif kalacakti.
  - Sonuc: X-ray-only seed42 mean balanced accuracy 0.919; seed7 0.902; seed123 0.852. Uc seed ortalamasi yaklasik 0.891.

- Frozen feature baseline denendi.
  - Neden: Fine-tuning'in gercekten katkisi var mi diye kontrol etmek.
  - Sonuc: Scratch/frozen ve pretrained/frozen sonuclar fine-tuned modelden dusuk kaldi; fine-tuning gerekli gorundu.

- Grad-CAM ve hata analizi olusturuldu.
  - Neden: Modelin neye baktigina ve hatalarin tipine kanit uretmek.
  - Sonuc: Seed42'de 42 validation hastasinin 37'si dogru, 5 hata false negative idi.

- Basit CT latent alignment denendi.
  - Neden: CT'deki 3D/anatomik bilgiyi X-ray encoder'a ogretmek.
  - Sonuc: Ilk raw CT projection alignment 0.886 balanced accuracy verdi; X-ray-only'den iyi olmadi.

- CT-mask latent alignment denendi.
  - Neden: Raw CT yerine sadece spine/mask anatomisine odaklanmak.
  - Sonuc: Ilk CT-mask alignment 0.902 verdi; raw CT'den iyi ama en iyi X-ray-only'den dusuktu.

- CT/DRR orientation ve HU threshold incelemesi yapildi.
  - Neden: Projection goruntuleri yatay/yanlis gorunuyordu ve kemik filtresi gerekiyordu.
  - Sonuc: Gorsel olarak en mantikli ayar axis=1, rot90 k=1, HU threshold=300 olarak secildi. Orijinal CT dosyalari degistirilmedi.

- HU300 corrected CT projection alignment egitildi.
  - Neden: Daha dogru DRR goruntusu skoru iyilestiriyor mu diye test etmek.
  - Sonuc: 0.886 balanced accuracy; gorsel kalite duzeldi ama skor artmadi.

- Corrected CT-mask alignment egitildi.
  - Neden: Maskelerin dogru orientation ile alignment'a daha iyi sinyal verip vermedigini olcmek.
  - Sonuc: 0.919 balanced accuracy; eski CT-mask sonucundan iyi ve X-ray-only seed42 ile ayni seviyede.

- Region-aware multitask classifier denendi.
  - Neden: Anomaly type yerine daha kaba ve daha gercekci bolge bilgisiyle modeli desteklemek.
  - Sonuc: Seed42 0.950, seed7 0.874; fakat region tahmini zayif kaldi. Bu nedenle bolge head'i guvenilir lokalizasyon degil, daha cok regularizer gibi yorumlanmali.

- Genel yorum:
  - En temiz baseline: X-ray-only pretrained ROI classifier.
  - En iyi tek seed sonucu: Region-aware multitask, ama region localization zayif.
  - En iyi CT branch: Corrected CT-mask alignment, X-ray-only seed42 ile ayni seviyede.
  - CT fikri hala degerli, fakat saglikli CT olmamasi, registration/FOV farki ve veri azligi nedeniyle mevcut basit alignment beklenen ekstra kazanci vermedi.
