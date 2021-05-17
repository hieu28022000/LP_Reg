import os 
import time

import numpy as np
import torch
import cv2
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


# def merge_bbox_in_line (bboxes, Y_DIST_FOR_MERGE_BBOX, EXPAND_FOR_BBOX):
#     for index, bbox in enumerate(bboxes):
#         try: 
#             if np.abs(bboxes[index][2][1] - bboxes[index-1][2][1]) < Y_DIST_FOR_MERGE_BBOX:
#                 bboxes[index][0], bboxes[index][1], bboxes[index][2], bboxes[index][3] = bboxes[index-1][0] - EXPAND_FOR_BBOX, bboxes[index][1] - EXPAND_FOR_BBOX, bboxes[index][2] + EXPAND_FOR_BBOX, bboxes[index-1][3] + EXPAND_FOR_BBOX
#                 print ('befor: ', bboxes)
#                 del_pos = index - 1
#                 bboxes = np.delete(bboxes, del_pos, axis=0)
#             print ('after: ', bboxes)
#         except: pass
#     return bboxes


def text_recog(cfg, opt, image_path, model, converter):
    text = 'None'
    if cfg.PIPELINE.DEEPTEXT:
        list_image_path = [image_path]
        for img in list_image_path:
            text = Deeptext_predict(img, opt, model, converter)
    elif cfg.PIPELINE.MORAN:
        text = MORAN_predict(cfg.PIPELINE.MORAN_MODEL_PATH, image_path, MORAN)
    return text

def text_detect_CRAFT(img, craft_config, CRAFT_MODEL, Y_DIST_FOR_MERGE_BBOX, EXPAND_FOR_BBOX, sortbb=True, visual_img=False):
    # img = loadImage(image_path)
    bboxes, polys, score_text = craft_text_detect(img, craft_config, CRAFT_MODEL)
    if sortbb:
        bboxes = sort_bb(bboxes)
    if visual_img:
        img = visual(img, polys)

    return bboxes, polys, score_text

def LP_detect_faster(img, cfg):
    classes = ['LP']
    outputs = predict_img_detectron2(cfg.FASTER_RCNN.MODEL, cfg.FASTER_RCNN.CONFIG, cfg.FASTER_RCNN.CONFIDENCE_THRESHOLD, cfg.FASTER_RCNN.NUM_OF_CLASS, img)
    frame = visualize (outputs, img, classes)
    cv2.imwrite('frame.jpg', frame)
    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes
    return boxes

def LP_detect_yolo(img, cfg, YOLO_NET):
    img, class_ids, boxes = yolo_detect(img, YOLO_NET, cfg)
    return boxes

def LP_regconition(cfg, img, YOLO_NET, img_path):
    
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
        bboxes, polys, score_text = text_detect_CRAFT(new_img, CRAFT_CONFIG, CRAFT_MODEL, PIPELINE_CFG.Y_DIST_FOR_MERGE_BBOX,  PIPELINE_CFG.EXPAND_FOR_BBOX)
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
    save = './result_reg'
    img_path = img_path.split('/')[-1]
    img_path = img_path.replace('.jpg', '.txt')
    path_save = os.path.join(save, img_path)
    # print (path_save)
    f = open(path_save, 'a')
    # print(name, confidence, xmin, ymin, xmax, ymax, img_path)
    f.write('{} {} {} {} {} {} {}'.format(name, str(confidence), str(xmin), str(ymin), str(xmax), str(ymax), '\n'))
    f.close()

if __name__ == '__main__':
    # start = time.time()
    # path = './data/reg_data'
    # save = './result_text_detect/'
    # detect_on_image(cfg, path)
    # for i in os.listdir(path):
    #     path_save = os.path.join(save, i)
    #     img_path = os.path.join(path, i)
    #     print (path_save)
    #     img = cv2.imread(img_path)cl
    #     bboxes, polys, score_text = text_detect_CRAFT(img, CRAFT_CONFIG, NET_CRAFT, PIPELINE_CFG.Y_DIST_FOR_MERGE_BBOX,  PIPELINE_CFG.EXPAND_FOR_BBOX)
    #     for i in bboxes:
    #         cv2.rectangle(img, (int (i[0][0]), int(i[0][1])), (int (i[2][0]), int(i[2][1])), (0,255,255), 1)
    #     cv2.imwrite(path_save, img)
    
    source = './evaluate_khang'
    for i in os.listdir(source):
        if (i.endswith('.jpg')):
            print (i)
            img_path = os.path.join(source, i)
            img = cv2.imread(img_path)
            img = LP_regconition(cfg, img, YOLO_NET, img_path)
            # cv2.imshow('image', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cv2.imwrite(os.path.join('result', i), img)
            


    # img = cv2.imread('un.png')    
    # bboxes, polys, score_text = text_detect_CRAFT(img, CRAFT_CONFIG, NET_CRAFT, PIPELINE_CFG.Y_DIST_FOR_MERGE_BBOX,  PIPELINE_CFG.EXPAND_FOR_BBOX)
    # print ('bboxes: ', len(bboxes))
    # for i in bboxes:
    #     print (i)
    #     cv2.rectangle(img, (int (i[0][0]), int(i[0][1])), (int (i[2][0]), int(i[2][1])), (255,0,0), 1)
    # cv2.imwrite('Khang.jpg', img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

#    text = text_recog (cfg, './result/LP.jpg', DEEPTEXT_MODEL, DEEPTEXT_PREDICTION, DEEPTEXT_CONVERTER)

