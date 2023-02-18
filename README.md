# YOLO v5 ve v7 Modelleri İçin Tahmin Alma

## YoloV7 prediction için iou hesaplama eklenecek
- [] Test dizininde yer alan görüntüleri ve .txt etiketlerini kullanır.
- [] Txt'de yer alan yolo formatındaki koordinatları normal koordinatlara çevirir.
- [] Modelden elde edilen tahmin görüntüsü ve etiketinin IoU skoru hesaplanır.
- [] IoU > 0 olan dosyaların sonlarında IoU skorları yer alır.
- [] IoU olmayanlarda "no_iou" olarak dosya isminin sonuna eklenir.

## YoloV5
- Test dizininde yer alan görüntüleri ve .txt etiketlerini kullanır.
- Txt'de yer alan yolo formatındaki koordinatları normal koordinatlara çevirir.
- Modelden elde edilen tahmin görüntüsü ve etiketinin IoU skoru hesaplanır.
- IoU > 0 olan dosyaların sonlarında IoU skorları yer alır.
- IoU olmayanlarda "no_iou" olarak dosya isminin sonuna eklenir.
