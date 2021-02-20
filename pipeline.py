import glob2
import cv2
import os 

import numpy as pd

from utils import get_config, loadImage, sorting_bounding_box, visual

from libs.CRAFT.craft import CRAFT
from libs.MORAN.MORAN_pred import MORAN_predict
from libs.MORAN.models.moran import MORAN
from libs.DeepText.Deeptext_pred import Deeptext_predict

from src import craft_text_detect

# setup config
cfg = get_config()
cfg.merge_from_file('configs/pipeline.yaml')
cfg.merge_from_file('configs/craft.yaml')

CRAFT_CONFIG = cfg.CRAFT
NET_CRAFT = CRAFT()

def text_recog(cfg, image_path):
    text = 'None'
    if cfg.PIPELINE.DEEPTEXT:
        list_image_path = [image_path]
        for img in list_image_path:
            text = Deeptext_predict(cfg.PIPELINE.DEEPTEXT_MODEL_PATH, img)
    elif cfg.PIPELINE.MORAN:
        text = MORAN_predict(cfg.PIPELINE.MORAN_MODEL_PATH, image_path, MORAN)
    
    return text

def text_detect_CRAFT(image_path, craft_config, net_craft, sortbb=True, visual_img=True):
    img = loadImage(image_path)
    bboxes, polys, score_text = craft_text_detect(img, craft_config, net_craft)

    if sortbb:
        polys = sorting_bounding_box(polys)
    if visual_img:
        img = visual(img, polys)
    
    return img, bboxes, polys, score_text

if __name__ == '__main__':
    # img = cv2.imread('data/Reg_data/00-A2_170.05.jpg')

    # predict region of bounding box
    # # img, bboxes, polys, score_text = text_detect_CRAFT('data/test.jpg', CRAFT_CONFIG, NET_CRAFT)

    # visualize bounding box
    # print ('bboxes--------',bboxes)
    # print ('score text---------', score_text)
    # print ('polys-------', polys)
    # count = 1
    # for i, line in enumerate(bboxes):
    #     print ('iiiiiiiiiiiii: ', line[0])
    #     cv2.rectangle(img, (line[0][0], line[0][1]), (line[2][0], line[2][1]), (0,255,0), 1)
    #     cv2.putText(img, str(count), (line[0][0], line[0][1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
    #     count += 1    
    # cv2.imwrite('img_re.jpg', img)

    # regconition a image
    text = text_recog (cfg, 'data/170.05.png')
    print ('text: ', text)
