# Kullanılan kütüphaneleri dahil etme
import cv2
import argparse
import numpy as np

# Komut satırı argümanlarını işlemek için argparse modülü kullanımı
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='D:/Apple-Detection/apple_tree_images')
ap.add_argument('-c', '--config', required=True,
                help='D:/Apple-Detection/yolov3.cfg')
ap.add_argument('-w', '--weights', required=True,
                help='D:/Apple-Detection/yolov3.weights')
ap.add_argument('-cl', '--classes', required=True,
                help='D:/Apple-Detection/yolov3.txt')
args = ap.parse_args()

# Ağın çıkış katmanlarının isimlerini almak için fonksiyon
def get_output_layers(net):
    layer_names = net.getUnconnectedOutLayersNames()
    return layer_names

# Tahmini çizmek için fonksiyon
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Görüntüyü yükle
image = cv2.imread(args.image)
if image is None:
    print("Görüntü yüklenirken hata oluştu.")
    exit()

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
classes = None

# Sınıf isimlerini yükle
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Rastgele renkler oluştur
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Ağı yükle
net = cv2.dnn.readNet(args.weights, args.config)

# Görüntüden blob oluştur
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

# Ağın çıkışlarını al
outs = net.forward(get_output_layers(net))

# Değişken ve dizi tanımlamalarını yapma
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.4
nms_threshold = 0.3
apple_count = 0

# Tahminlerin her birini işlemek için döngü
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

# Tahminlerin üst üste binmesini engellemek için NMS kullan
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Tahminleri çiz
if len(indices) > 0:
    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        apple_count += 1
else:
    print("Fotoğrafta Elma tespit edilmedi.")

# Elma sayısını yazdır
print(f"Fotoğraftaki Toplam Elma Sayısı: {apple_count}")

# Sonucu pencerede göster
cv2.imshow("Elma Tespiti Penceresi", image)
cv2.waitKey()

# Sonucu kaydet
cv2.imwrite("result.jpg", image)
cv2.destroyAllWindows()
