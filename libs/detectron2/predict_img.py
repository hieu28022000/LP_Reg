import os
import cv2
import json
import random
import itertools
import numpy as np

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, evaluator
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt

import json

def predict_img_detectron2 (path_weigths, path_config, confidence_threshold, num_of_class, img):
  cfg = get_cfg()
  cfg.MODEL.DEVICE='cpu'
  cfg.merge_from_file(path_config)
  cfg.MODEL.WEIGHTS = path_weigths

  #cfg.MODEL.WEIGHTS = "mask_rcnn_R_50_FPN_3x_model/model_final.pth"
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8   
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class 
  predictor = DefaultPredictor(cfg)
  outputs = predictor(img)
  
  return outputs

# Đầu vào detect = output của hàm predict, frame = original image của mình, classs = tên class để visualize
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
      cv2.rectangle(frame, start, end, (random.randint(0,255),random.randint(0,255),255), 1)
      cv2.putText(frame, str (classs[color]),start, cv2.FONT_HERSHEY_PLAIN, 1, (random.randint(0,255),random.randint(0,255),255), 2)
  return frame

  # def main:
  #   path_weigth = 
  #   path_config =
  #   confidences_threshold = 
  #   num_of_class = 
  #   path_img = 
  #   classes = ['LP']
  #   _frame = cv2.imread(path_img)
  #   outputs = predict(cfg.FASTER_RCNN.MODEL, cfg.FASTER_RCNN.MODEL.CONFIG, cfg.FASTER_RCNN.CONFIDENCE_THRESHOLD, cfg.FASTER_RCNN.NUM_OF_CLASS, './')
  #   frame = visualize (outputs, _frame, classes )
  #   cv2_imshow(frame)
