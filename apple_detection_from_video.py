# Kullanılan kütüphaneler dahil etme
import cv2
import argparse
import numpy as np

# Argüman analizi
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

def get_output_layers(net):
    layer_names = net.getUnconnectedOutLayersNames()
    return layer_names

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label + " " + str(round(confidence, 2)), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Sınıfları yükle
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Her bir sınıf için rastgele renkler oluştur
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Sinir ağını yükle
net = cv2.dnn.readNet(args.weights, args.config)

# Videoyu yükle
video = cv2.VideoCapture(args.video)

while True:
    ret, frame = video.read()
    if not ret:
        break

    Genişlik = frame.shape[1]
    Yükseklik = frame.shape[0]
    ölçek = 0.00392

    blob = cv2.dnn.blobFromImage(frame, ölçek, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4
    nms_threshold = 0.3

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                merkez_x = int(detection[0] * Genişlik)
                merkez_y = int(detection[1] * Yükseklik)
                w = int(detection[2] * Genişlik)
                h = int(detection[3] * Yükseklik)
                x = merkez_x - w // 2
                y = merkez_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices:
            kutu = boxes[i]  # İndeks doğrudan kullanılarak kutuya eriş
            x, y, w, h = kutu[0], kutu[1], kutu[2], kutu[3]
            draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    else:
        print("Elma algılanmadı")

    cv2.imshow("nesne tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video yakalama ve pencereleri kapat
video.release()
cv2.destroyAllWindows()
