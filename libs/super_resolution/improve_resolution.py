import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN
import cv2

# img = Image.open('../../data/cool.png')
def improve_resolution(lr_img, model):
    # save_img = cv2.resize(lr_img, (500,500))
    # cv2.imwrite('./before.jpg', save_img)
    # lr_img = np.array(img)

    # model = RRDN(weights='gans')
    sr_img = model.predict(lr_img)
    return sr_img
    # print (sr_img.shape)
    # sr_img = cv2.resize(sr_img, (500,500))
    # # Image.fromarray(sr_img)
    # cv2.imwrite('./after.jpg', sr_img)
