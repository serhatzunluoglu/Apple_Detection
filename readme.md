# Projeyi Çalıştırmak için

Projeyi clone'ladıktan sonra bu [link](https://drive.google.com/drive/folders/1EQ_-72dGQIOrpN2LXOFmhGFQCWEUwrll)  üzerinden yolov3.weights dosyasını indirin ve Apple_Detection klasörünün içine ekleyin.

## Fotoğraflar için Terminal Komutu

```bash
python apple_detection_from_photo.py -i apple_tree_images_and_vids/apple-1.jpg -c yolov3.cfg -cl yolov3.txt -w yolov3.weights
```

## Videolar için Terminal Komutu

```bash
python apple_detection_from_video.py -v apple_tree_images_and_vids/apple_video_1.mp4 -c yolov3.cfg -cl yolov3.txt -w yolov3.weights
```

### Fotoğraf Programının Ekran Görüntüsü
![Fotoğraf çıktısı](https://github.com/serhatzunluoglu/Apple_Detection/blob/032eeefd972686ea2969f7523ea3c12e3092b159/result.jpg)

### Video Programının Ekran Görüntüsü
![Video çıktısı](https://github.com/serhatzunluoglu/Apple_Detection/blob/66c8d8bf3be5e3e80fefff9306830dbc61ed56c9/Video%20Program.gif)
