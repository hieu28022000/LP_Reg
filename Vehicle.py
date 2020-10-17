import cv2
import numpy as np
import os
import random
import imutils
import time
import glob


classes = ['License plate']
# classesFile = "obj.names"
# classes = None
# with open(classesFile, 'rt') as f:
#     classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNet("yolov3_BS_last.weights", "yolov3_BS.cfg")
output_path = os.path.join("output", "out_img.jpg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect_image(img):

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

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            # cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            # cv2.putText(img, label, (x, y-2), font, 1, color, 2)
    
    # output_image = img[y:(y+h), x:(x+w), :]
    # Store image
    # cv2.imwrite(output_path, img)   

    return img, output_path, x, y, x+w, y+h

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = cap.read()
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter("output/Vid78.avi",codec,15,(WIDTH,HEIGHT))
    cap.release()
    counts = 0
    cap = cv2.VideoCapture(video_path)
    while (True):
        ret, frame = cap.read() 
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-2), font, 1, color, 2)
    
        counts += 1
        # cv2.imshow('detection', frame)
        writer.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def Crop_Vehicle(img, x_min, y_min, x_max, y_max):
    
    # print (x_min,y_min,x_max,y_max)
    Cropped_img = img[y_min:y_max+1, x_min:x_max+1]

    return Cropped_img

def Get_color(img_Cropped):

    img_gray = cv2.cvtColor(img_Cropped, cv2.COLOR_BGR2GRAY)
    ret, bw_img = cv2.threshold (img_gray, 127,255, cv2.THRESH_BINARY)

    black = 0
    white = 0
    x_black_pixel = []
    y_black_pixel = []
    x_white_pixel = []
    y_white_pixel = []
    for x in range(0, bw_img.shape[1], 8):
        for y in range(0, bw_img.shape[0], 8):
            if bw_img[y,x] == 255:
                white += 1
                x_white_pixel.append(x)
                y_white_pixel.append(y)
            else:
                black +=1
                x_black_pixel.append(x)
                y_black_pixel.append(y)
    
    Blue_Sum = 0
    Green_Sum = 0
    Red_Sum = 0
    if max(black, white) == white:
        for pixel in range(len(x_white_pixel)):
            Blue_Sum += img_Cropped[y_white_pixel[pixel], x_white_pixel[pixel], 0]
            Green_Sum += img_Cropped[y_white_pixel[pixel], x_white_pixel[pixel], 1]
            Red_Sum += img_Cropped[y_white_pixel[pixel], x_white_pixel[pixel], 2]
        Blue_average = Blue_Sum // white
        Green_average = Green_Sum // white
        Red_average = Red_Sum // white

    elif max(black, white) == black:
        for pixel in range(len(x_black_pixel)):
            Blue_Sum += img_Cropped[y_black_pixel[pixel], x_black_pixel[pixel], 0]
            Green_Sum += img_Cropped[y_black_pixel[pixel], x_black_pixel[pixel], 1]
            Red_Sum += img_Cropped[y_black_pixel[pixel], x_black_pixel[pixel], 2]
        Blue_average = Blue_Sum // black
        Green_average = Green_Sum // black
        Red_average = Red_Sum // black

    return Blue_average, Green_average, Red_average

def Create_LP(img_Cropped, Blue, Green, Red):
    
    height, width, channels = img_Cropped.shape
    w = width * 5
    h = height * 5
    LP_img = np.zeros((h,w,3), dtype='uint8')
    for x in range(w):
        for y in range(h): 
            LP_img[y][x][0] = Blue
            LP_img[y][x][1] = Green
            LP_img[y][x][2] = Red
    
    xmin_LP = width * 2
    xmax_LP = width * 3
    ymin_LP = height * 2
    ymax_LP = height * 3

    LP_img[ymin_LP:ymax_LP, xmin_LP:xmax_LP,:] = img_Cropped[:,:,:] 

    return LP_img


if __name__ == '__main__':
    # if not os.path.exists("output"):
    #     os.mkdir("output")

    # # Detect video
    # detect_video('input/IMG_E5177.MOV')

    # Detec image

    image = cv2.imread('images/17MD788888.png')
    img_detected, image_path, x, y , x_max, y_max = detect_image(image)
    img_Cropped = Crop_Vehicle(image, x, y, x_max, y_max)
    Blue, Green, Red = Get_color(img_Cropped)
    new_LP = Create_LP(img_Cropped, Blue,  Green, Red)



    cv2.imshow('Normal', image)
    cv2.imshow('Cropped', img_Cropped)
    cv2.imshow('New License plate', new_LP)
    cv2.waitKey(0)