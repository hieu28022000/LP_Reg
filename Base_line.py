import cv2
import numpy as np
import glob
import os
import random
import imutils
import time
import pytesseract
import numba

from timeit import default_timer as timer 
from PIL import Image
from Vehicle import detect_image, detect_video, Crop_Vehicle, Create_LP, Get_color
from Text import Get_text
from numba import cuda

def Base_line(img):
    # Vehicle
    net = cv2.dnn.readNet("yolov3_BS_last.weights", "yolov3_BS.cfg")
    output_path = os.path.join("out_img.jpg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    classes = ['License plate']

    # Text
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Vehicle detect
    image, image_path, x_min, y_min, x_max, y_max = detect_image(img)
    # cv2.imwrite('License_Plates/17MD788888.jpg', image[y_min:y_max+1][x_min:x_max+1])

    # Crop license plate image
    LP_Cropped = Crop_Vehicle(image, x_min, y_min, x_max, y_max)

    # Get color average in license plate
    Blue, Green, Red = Get_color(LP_Cropped)

    # Create new image license plate
    LP_img = Create_LP(LP_Cropped, Blue, Green, Red)
    cv2.imwrite('License_Plates/License_Plates.jpg', LP_img)

    # Get text from crop vehicle image
    text, image_path = Get_text('License_Plates/License_Plates.jpg')

    # Visual
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,255,0), 2)
    cv2.putText(image, str(text), (x_min, y_min-2), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    cv2.imshow('License plate', image)
    cv2.waitKey(0)


if __name__ == '__main__':

    if not os.path.exists('Demo'):
        os.makedirs('Demo')
    name = 'test2.jpg'
    image = cv2.imread('images/'+ name)
    cuda.select_device(0)
    Base_line(image)
    cuda.close()

    cv2.imwrite('Demo/' + name, image)    