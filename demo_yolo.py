from utils import get_config
import cv2
from src import yolo_detect

cfg = get_config()
cfg.merge_from_file('configs/yolo.yaml')

yolo_net = cv2.dnn.readNet(cfg.YOLOV4.YOLO_MODE)
img = cv2.imread('./data/a_164337.jpg')
yolo_detect(img, net, cfg)
