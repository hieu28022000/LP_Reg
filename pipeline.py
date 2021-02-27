import cv2
import os 
import time

import numpy as np

from utils import get_config, loadImage, sorting_bounding_box, visual

from libs.CRAFT.craft import CRAFT
from libs.MORAN.MORAN_pred import MORAN_predict
from libs.MORAN.models.moran import MORAN
from libs.DeepText.Deeptext_pred import Deeptext_predict
from libs.detectron2.predict_img import predict_img_detectron2
from libs.detectron2.predict_img import visualize

from src import craft_text_detect

# setup config
cfg = get_config()
cfg.merge_from_file('configs/pipeline.yaml')
cfg.merge_from_file('configs/craft.yaml')
cfg.merge_from_file('configs/faster.yaml')

CRAFT_CONFIG = cfg.CRAFT
NET_CRAFT = CRAFT()

if os.path.exits('./result') == False:
    os.mkdir('./result')

if os.path.exists('./models') == False:
    os.mkdir('./models')

def text_recog(cfg, image_path):
    text = 'None'
    if cfg.PIPELINE.DEEPTEXT:
        list_image_path = [image_path]
        for img in list_image_path:
            text = Deeptext_predict(cfg.PIPELINE.DEEPTEXT_MODEL_PATH, img)
    elif cfg.PIPELINE.MORAN:
        text = MORAN_predict(cfg.PIPELINE.MORAN_MODEL_PATH, image_path, MORAN)
    
    return text

def text_detect_CRAFT(img, craft_config, net_craft, sortbb=True, visual_img=False):
    # img = loadImage(image_path)
    bboxes, polys, score_text = craft_text_detect(img, craft_config, net_craft)

    if sortbb:
        polys = sorting_bounding_box(polys)
    if visual_img:
        img = visual(img, polys)
    
    return bboxes, polys, score_text

def LP_detect_faster(img, cfg):
    classes = ['LP']
    outputs = predict_img_detectron2(cfg.FASTER_RCNN.MODEL, cfg.FASTER_RCNN.CONFIG, cfg.FASTER_RCNN.CONFIDENCE_THRESHOLD, cfg.FASTER_RCNN.NUM_OF_CLASS, img)
    frame = visualize (outputs, img, classes)
    cv2.imwrite('frame.jpg', frame)
    print (outputs)
    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes
    return boxes


if __name__ == '__main__':
    start = time.time()
    img = cv2.imread('data/a_164337.jpg')

    # detect License plates in image    
    detected_LP = LP_detect_faster(img, cfg)
    for i in detected_LP:
        # store the license plate in image to new_img variable
        print (i)
        new_img = img[int(i[1]):int(i[3]), int(i[0]):int(i[2])]

        # predict region of bounding box
        bboxes, polys, score_text = text_detect_CRAFT(new_img, CRAFT_CONFIG, NET_CRAFT)
        count = 1
        LP_reg = []
        for j, bbox in enumerate(bboxes):
            img_reg = new_img[ int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0])]   
            cv2.imwrite('./reg/img_reg.jpg', img_reg)
            text = text_recog (cfg, './reg/img_reg.jpg')
            LP_reg.append(text)
            # cv2.rectangle(new_img, (bbox[0][0], bbox[0][1]), (bbox[2][0], bbox[2][1]), (0,255,0), 1)
            # cv2.putText(new_img, str(count), (bbox[0][0], bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
        LP_reg_text = ''.join(LP_reg)
        print (int(i[0]), int(i[1]))
        cv2.putText(img, str(LP_reg_text), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=1)
    cv2.imwrite('img_re.jpg', img)

        # regconition a image
        # text = text_recog (cfg, 'data/170.05.png')
        # print ('text: ', text)
    end = time.time()
    print ('time: ', end - start)
