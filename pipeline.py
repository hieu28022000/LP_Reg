'''
author: Khang Nguyen Huu
created: 3/2021
last modified: 8/4/2021
'''
import os 
import time

import numpy as np
import torch
import cv2
import argparse
from ISR.models import RDN, RRDN

from utils import get_config, loadImage, sorting_bounding_box, visual, align_item, tlwh_2_maxmin, merge_bb, four_point_transform, sort_bb

from libs.CRAFT.craft import CRAFT
from libs.MORAN.MORAN_pred import MORAN_predict
from libs.MORAN.models.moran import MORAN
from libs.DeepText.Deeptext_pred import Deeptext_predict, load_model_Deeptext
#from libs.detectron2.predict_img import predict_img_detectron2
#from libs.detectron2.predict_img import visualize
from libs.super_resolution.improve_resolution import improve_resolution

from src import craft_text_detect, load_model_Craft
from src import yolo_detect
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup config
cfg = get_config()
cfg.merge_from_file('configs/pipeline.yaml')
cfg.merge_from_file('configs/craft.yaml')
cfg.merge_from_file('configs/faster.yaml')
cfg.merge_from_file('configs/yolo.yaml')
cfg.merge_from_file('configs/Deeptext.yaml')

DEEPTEXT_CONFIG = cfg.DEEPTEXT
CRAFT_CONFIG = cfg.CRAFT
NET_CRAFT = CRAFT()
PIPELINE_CFG = cfg.PIPELINE

# load all model
# model yolo
print ('[LOADING] Detect model')
YOLO_NET = cv2.dnn.readNet(cfg.YOLOV4.YOLO_MODEL_PATH, cfg.YOLOV4.YOLO_CFG_PATH)
print ('[LOADING SUCESS] Detect model')
# model text detct
print ('[LOADING] Text detecttion model')
CRAFT_MODEL = load_model_Craft(CRAFT_CONFIG, NET_CRAFT)
print ('[LOADING SUCESS] Text detection model')
# model regconition
print ('[LOADING] Text regconition model')
DEEPTEXT_MODEL, DEEPTEXT_CONVERTER = load_model_Deeptext(DEEPTEXT_CONFIG)
print ('[LOADING SUCESS] Text regconition model')
print ('[LOADING] Super resolution model')
super_resolution_model = RRDN(weights='gans')
print ('[LOADING SUCESS] Super resolution model')

def text_recog(cfg, opt, image_path, model, converter):
    text = 'None'
    if cfg.PIPELINE.DEEPTEXT:
        list_image_path = [image_path]
        for img in list_image_path:
            text = Deeptext_predict(img, opt, model, converter)
    elif cfg.PIPELINE.MORAN:
        text = MORAN_predict(cfg.PIPELINE.MORAN_MODEL_PATH, image_path, MORAN)
    return text

def text_detect_CRAFT(img, craft_config, CRAFT_MODEL, sortbb=True, visual_img=False):
    '''
    args:
        img: image
        craft_config: config of craft
        CRAFT_MODEL: craft model
        sort_bb: whether or not sort bounding box
        visual_image: whether or no not visual image
    return:
        bboxes: bbox of text
        polys: polygon of text
        score_text: confidence score
    '''
    # img = loadImage(image_path)
    bboxes, polys, score_text = craft_text_detect(img, craft_config, CRAFT_MODEL)
    if sortbb:
        bboxes = sort_bb(bboxes)
    if visual_img:
        img = visual(img, polys)

    return bboxes, polys, score_text

def LP_detect_yolo(img, cfg, YOLO_NET):
    '''
    Localize the license plate in image
    args:
        img
        cfg: yolo config
        YOLO_NET
    return:
        boxes: list of bounding box license plate in image
    '''
    img, class_ids, boxes = yolo_detect(img, YOLO_NET, cfg)
    return boxes

def LP_regconition(cfg, img, YOLO_NET):
    '''
    main function to do license plate recognition in image, this will loop 
    over license plate candidates in image and do recognition with each
    args:
        cfg: full config
        img: image to recognition
        YOLO_NET: model yolo
    '''
    
    # detect License plates in image    
    detected_LP = LP_detect_yolo(img, cfg, YOLO_NET)
    for i in detected_LP:
        # store the license plate in image to new_img variable
        print ("detected license plates: ", i)
        if i[0] < 0: i[0] = 0
        if i[1] < 0: i[1] = 0
        if i[2] < 0: i[2] = 0
        if i[3] < 0: i[3] = 0
        new_img = img[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
        cv2.imwrite('./result/LP.jpg', new_img)

        # predict region of text bounding box
        bboxes, polys, score_text = text_detect_CRAFT(new_img, CRAFT_CONFIG, CRAFT_MODEL)
        LP_reg = []
        # count = 1
        for index, bbox in enumerate(bboxes):
            # merge bbox on a line
            if bbox[0][0] < 0: bbox[0][0] = 0
            if bbox[0][1] < 0: bbox[0][1] = 0
            if bbox[1][0] < 0: bbox[1][0] = 0
            if bbox[1][1] < 0: bbox[1][1] = 0
            img_reg = new_img[int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0])]
            img_reg = improve_resolution(img_reg, super_resolution_model)
            cv2.imwrite('./reg/img_reg.jpg', img_reg)
            text = text_recog(cfg, DEEPTEXT_CONFIG, './reg/img_reg.jpg', DEEPTEXT_MODEL, DEEPTEXT_CONVERTER)
            # text = text_recog (cfg, './reg/img_reg.jpg', DEEPTEXT_MODEL, DEEPTEXT_PREDICTION, DEEPTEXT_CONVERTER)
            LP_reg.append(text)
            # cv2.rectangle(new_img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0,255,0), 1)
            # cv2.putText(new_img, str(count), (int(bbox[0][0]), int(bbox[0][1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
            # count += 1
        LP_reg_text = ''.join(LP_reg)
        LP_reg_text = LP_reg_text.upper()
        print (LP_reg)
        write_predict(LP_reg_text, '1', int(i[0]), int(i[1]), int(i[2]), int(i[3]) , img_path)
        cv2.putText(img, str(LP_reg_text), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,255,0), thickness=3)
    return img
    

def write_predict(name, confidence, xmin, ymin, xmax, ymax, img_path):
    '''
    function to write the output of license plate recognition in txt form
    '''
    save = './result_reg'
    img_path = img_path.split('/')[-1]
    img_path = img_path.replace('.jpg', '.txt')
    path_save = os.path.join(save, img_path)
    # print (path_save)
    f = open(path_save, 'a')
    # print(name, confidence, xmin, ymin, xmax, ymax, img_path)
    f.write('{} {} {} {} {} {} {}'.format(name, str(confidence), str(xmin), str(ymin), str(xmax), str(ymax), '\n'))
    f.close()

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_on_folder', type=bool, default=False,
                        help='Wheter or not run on folder')
    parser.add_argument("-i", "--image_path", type=str,
                        help='Path to image')
    parser.add_argument("--folder_path", type=str, default='./data',
                        help='Path to folder')
    parser.add_argument('--save_image_folder', type=str, default='./visualize_output',
                        help='Path to save visualize image')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if os.path.exists(args.save_image_folder == False):
        print("Not found save folder, create one with name {}".format(args.save_image_folder))
        os.mkdir(args.save_image_folder)

    if (args.run_on_folder == True):
        for i in os.listdir(args.folder_path):
            if (i.endswith('.jpg')):
                img_path = os.path.join(args.folder_path, i)
                img = cv2.imread(img_path)
                img = LP_regconition(cfg, img, YOLO_NET, img_path)
                cv2.imwrite(os.path.join('result', i), img)
    else:
        img = cv2.imread(args.image_name)
        img = LP_regconition(cfg, img, YOLO_NET, img_path)
        cv2.imwrite(os.path.join('result', i), img)

            