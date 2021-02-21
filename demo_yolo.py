import cv2
import numpy as np
import os
from utils import get_config


cfg = get_config()
cfg.merge_from_file('configs/yolov4.yaml')

net = cv2.dnn.readNet(cfg.YOLO.YOLO_MODEL_PATH, cfg.YOLO.YOLO_CFG_PATH)
output_path = os.path.join("results", "out_img.jpg")
# Name custom object;
classesFile = cfg.YOLO.CLASS_PATH;

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



def detect(img, net, output_layers):

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x_min = int(center_x - w / 2)
                y_min = int(center_y - h / 2)
                x_max = x_min + w
                y_max = y_min + h
                boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x, y), color, 2)
            cv2.putText(img, label, (x, y-2), font, 1, color, 2)  

    return img, class_ids, boxes


# detect(frame, net, output_layers)




