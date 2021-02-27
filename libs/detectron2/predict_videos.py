import os
import cv2
import json
import random
import itertools
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
# from detectron2.engine import DefaultPredictor
# #from detectron2.detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.config import get_cfg
# #from detectron2.utils.visualizer import Visualizer, ColorMode
# #from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
# from detectron2.structures import BoxMode

# import json

# cfg = get_cfg()
# cfg.MODEL.DEVICE='cpu'
# '''Load Faster RCNN
# cfg.MODEL.WEIGHTS = "./model/faster_rcnn_R_101_FPN_3x_model/model_final.pth"'''
# # --- #
# # Load Mask RCNN configuration
# cfg.merge_from_file("detectron/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.MODEL.WEIGHTS = "model/mask_rcnn_R_50_FPN_3x_model/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95 # Correct class result must be more than 95%
# predictor = DefaultPredictor(cfg)


from detectron2.engine import DefaultPredictor
#from detectron2.detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer, ColorMode
#from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import imutils
import json

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'

# cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file("/content/drive/My Drive/Khang làm ở chỗ này/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
cfg.MODEL.WEIGHTS = "/content/drive/My Drive/Khang làm ở chỗ này/Model_AI_challenge/Detectron/model_fasterRCNN_Khang_10k _46_dataupdate.pth"
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
print ("[Loading] Model sucessful")
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Correct class result must be more than 50%
predictor = DefaultPredictor(cfg)
#classs = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck','boat']
classs = ['Loai 1', 'Loai 2', 'Loai 3', 'Loai 4', 'Loai 5']
#classs = ['di_bo','xe_dap','xe_may','xe_hang_rong','xe_ba_gac','xe_taxi','xe_hoi','xe_ban_tai','xe_cuu_thuong','xe_khach','xe_buyt','xe_tai','xe_container','xe_cuu_hoa']

def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels

def object_detect(image):
    predictions = predictor(image)
    boxes = predictions["instances"].pred_boxes 
    scores = predictions["instances"].scores 
    classes = predictions["instances"].pred_classes 
    # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
    return predictions

    

def register (classes, dicts):
    for i in range(len(dicts)):
        for j in range(len(dicts[i]["annotations"])):
            dicts[i]["annotations"][j]['bbox_mode'] = BoxMode.XYXY_ABS
    data = [dicts]

    for index, d in enumerate(["train"]):
        DatasetCatalog.register("fptt/" + d, lambda index=index: data[index])
        MetadataCatalog.get("fptt/" + d).set(thing_classes=classes)
    Metadata = MetadataCatalog.get("fptt/train")
    return Metadata

def visualize (out, frame, classs):
  boxes = out['instances'].pred_boxes
  scores = out['instances'].scores
  classes = out['instances'].pred_classes
  for i in range (len(classes)):
    if (scores[i] > 0.5):
      for j in boxes[i]:
        start = (int (j[0]), int (j[1]))
        end = (int (j[2]), int (j[3]))
      color = int (classes[i])
      print (classes[i])
      cv2.rectangle(frame, start, end, (random.randint(0,255),random.randint(0,255),255), 1)
      cv2.putText(frame, str (classs[color]),start, cv2.FONT_HERSHEY_PLAIN, 1, (random.randint(0,255),random.randint(0,255),255), 2)
  return frame

  def convert(size, box):
    re = []
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw #x center
    w = w*dw
    y = y*dh #y center
    h = h*dh
    x = str (x)
    y = str (y)
    w = str (w)
    h = str (h)
    re.append (x)
    re.append (y)
    re.append (w)
    re.append (h)
    return re

def write_txt_detectron(file_name, obj_detect, size, output_path):
  boxes = obj_detect['instances'].pred_boxes
  scores = obj_detect['instances'].scores
  classes = obj_detect['instances'].pred_classes
  f = open(output_path + '/' + file_name + '.txt', 'w')
  _scores = []
  _classes = []
  for i in scores:
    i = float (i)
    i = str (i)
    _scores.append(i)
  for i in classes:
    i = int (i)
    i = str (i)
    _classes.append (i)
  final = []
  tmp = []
  for i in range (len(boxes.tensor)):
    for j in boxes.tensor[i]:
      j = float (j)
      #j = str (j)
      tmp.append (j)
      if (len(tmp) == 4):
        tmp = convert(size, tmp)
        tmp.append(_scores[i])
        tmp.append (_classes[i])
        final = tmp
        tmp = []
        final = ' '.join(final)
        f.write(final)
        f.write('\n')
        final = []

  f.close()

    
if __name__ == '__main__':
    #classes = ['di_bo','xe_dap','xe_may','xe_hang_rong','xe_ba_gac','xe_taxi','xe_hoi','xe_ban_tai','xe_cuu_thuong','xe_khach','xe_buyt','xe_tai','xe_container','xe_cuu_hoa']
    #register data
    # with open('/content/drive/My Drive/Khang làm ở chỗ này/frcnn_train_dicts.json', 'r') as fp:
    #   train_dicts = json.load(fp)
    # Metadata = register(classes, train_dicts)
    output_dir = '/content/output'
    cap = cv2.VideoCapture('/content/drive/My Drive/Counting/aic-hcmc2020/videos/cam_18.mp4')
    #cap = cv2.VideoCapture ('/content/drive/My Drive/Counting/Video/Videos/NKKN-VoThiSau 2017-07-18_08_00_00_000.asf')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc =  cv2.VideoWriter_fourcc(*'XVID')
    size = (int (cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter( output_dir + '/cam_18.avi', fourcc, 15, size, 1)
    
    count = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if ret != True:
            break
        outputs = object_detect(frame)
        video = visualize(outputs, frame, classs)
        #video = cv2.resize(video, size)
        #print (video.shape)
        out.write(video)
        frame_id += 1
        print ('process on frame: ', frame_id)
        write_txt_detectron ('frame_' + str(frame_id), outputs, size, output_dir)
        key = cv2.waitKey(1)  & 0xff
        if key == 27:
            break
    
    out.release()
    cap.release()

