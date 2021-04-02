import cv2
import numpy as np

def nms(bounding_boxes, confidence_score, threshold):
    if len(bounding_boxes) == 0:
        return [], []
    boxes = np.array(bounding_boxes)

    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2] + start_x 
    end_y = boxes[:, 3] + start_y 

    score = np.array(confidence_score)

    picked_boxes = []
    picked_score = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    order = np.argsort(score)

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def yolo_detect(img, net, config):

    # Name custom object;
    classesFile = config.YOLOV4.CLASS_PATH

    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


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

    boxes, confidences = nms(boxes, confidences, 0.3)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    new_boxes = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            new_boxes.append([x,y,w,h])
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (w, h), color, 2)
            cv2.putText(img, label, (x, y-2), font, 1, color, 2)  

    return img, class_ids, new_boxes

if __name__ == '__main__':
    detect(frame, net, output_layers)



