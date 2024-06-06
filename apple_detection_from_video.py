# Kullanılan kütüphaneleri dahil etme
import cv2
import argparse
import numpy as np

# Komut satırı argümanlarını işlemek için argparse modülü kullanımı
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help='D:/Apple_Detection/apple_tree_images_and_vids/apple_garden.mp4')
ap.add_argument('-c', '--config', required=True,
                help='D:/Apple_Detection/yolov3.cfg')
ap.add_argument('-w', '--weights', required=True,
                help='D:/Apple_Detection/yolov3.weights')
ap.add_argument('-cl', '--classes', required=True,
                help='D:/Apple_Detection/yolov3.txt')
args = ap.parse_args()

# YOLO modelinden çıktı katmanlarının isimlerini alır
def get_output_layers(net):
    layer_names = net.getUnconnectedOutLayersNames()
    return layer_names

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # Algılanan nesnenin sınıf adını alır
    label = str(classes[class_id])
    # Algılanan nesnenin rengini belirler
    color = COLORS[class_id]

    # Algılanan nesnenin etrafına çerçeve çizer
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # Dikdörtgenin içine sınıf adını ve güven skorunu yazar
    cv2.putText(img, label + " " + str(round(confidence, 2)), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Sınıfları yükle
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Her seferinde çerçeve için rastgele renkler oluştur
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Sinir ağını yükle
net = cv2.dnn.readNet(args.weights, args.config)

# Videoyu yükle
video = cv2.VideoCapture(args.video)

while True:
    # Videodan bir kare al ve eğer kare alınamazsa döngüyü kır
    ret, frame = video.read()
    if not ret:
        break

    # Kare genişliğini ve yüksekliğini al
    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    # YOLO modeline uygun blob oluştur
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # YOLO ile tespit yap
    outs = net.forward(get_output_layers(net))

    # Değişken ve dizi tanımlamaları
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4
    nms_threshold = 0.3

    # YOLO modelinin çıktılarını işleyerek tespit edilen nesnelerin bilgilerini topla
    # Algılanan nesneler için güven skoru belirli bir eşik değerinden büyükse, sınıf ID'si ve güven skoru listelere eklenir
    # Sonuç olarak, tespit edilen nesnelerin sınıf ID'leri, güven skorları ve sınırlayıcı kutu koordinatları listelenir
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression (NMS) uygular(birden çok kez algılanan nesneler arasından en uygun olanı seçer).
    # İndeksleri, kutuların koordinatları ve güven skorlarına dayanarak belirlenen eşik değerleriyle NMS algoritmasına gönderir.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


    # Algılanan nesneleri çerçevelerle işaretleme
    if len(indices) > 0:
        for i in indices:
            kutu = boxes[i]
            x, y, w, h = kutu[0], kutu[1], kutu[2], kutu[3]
            draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    else:
        print("Elma algılanmadı.")


    # Video kodunu çalıştırdıktan sonra esc tuşuna basarak durdur
    cv2.imshow("Video Penceresi", frame)
    if cv2.waitKey(1) == 27:
        break

# Video yakalama ve pencereleri kapat
video.release()
cv2.destroyAllWindows()
