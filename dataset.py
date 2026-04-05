# Dataloader for NIfTI/PNG medical images and segmentation masks.
# Veriler kendi aralarında tutarlı olmalı(Xray çözünürlükleri hepsi aynı, ct aynı gibisinden)
# Maske yok(Sağlıklı hasta) ise boyama yapmamayı öğretmek için aynı çözünürlükte 0 değerli bir görsel oluşturmak
# Hasta ve maskesizse o datayı silmek
# Model başarılarına göre filtrelemeler
# Data klasör yapısı data -> patient001 -> ct.nii.gz , mr.nii.gz , xray.png , mask.png , ct_mask.nii.gz , mr_mask.nii.gz 