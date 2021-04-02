import cv2
import numpy as np 

def loadImage(img_file):
    img = cv2.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def tlwh_2_maxmin(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        new_bboxes.append([xmin, ymin, xmax, ymax])
    new_bboxes = np.array(new_bboxes)
    return new_bboxes