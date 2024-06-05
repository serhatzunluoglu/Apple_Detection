# Projeyi Çalıştırmak için

Projeyi clone'ladıktan sonra bu [link](https://drive.google.com/drive/folders/1EQ_-72dGQIOrpN2LXOFmhGFQCWEUwrll)  üzerinden yoloV3 dosyasını indirin Apple_Detection klasörünün içine ekleyin.


## Fotoğraflar için terminal komutu

```bash
python apple_detection_from_photo.py -i apple_tree_images_and_vids/apple-1.jpg -c yolov3.cfg -cl yolov3.txt -w yolov3.weights
```

## Videolar için terminal komutu

```bash
python apple_detection_from_video.py -v apple_tree_images_and_vids/apple_video_1.mp4 -c yolov3.cfg -cl yolov3.txt -w yolov3.weights
```
