import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *


# Load model
conf_thres = 0.4
iou_thres = 0.5
prob_thres = 0.7
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
imgsz = (1280,1280)
cfg = './cfg/yolor_p6_score.cfg'
weights = "./runs/train/yolor_p6/weights/best_overall.pt"
img_path = "./test"
names = ["QP","NY","QG"]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

model = Darknet(cfg, imgsz).cuda()
model.load_state_dict(torch.load(weights, map_location=device)['model'])
# model = attempt_load(weights, map_location=device)  # load FP32 model
#imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
model.to(device).eval()
# if half:
#     model.half()  # to FP16

# # Second-stage classifier
# classify = False
# if classify:
#     modelc = load_classifier(name='resnet101', n=2)  # initialize
#     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#     modelc.to(device).eval()

files = os.listdir(img_path)
for file in files:
    path = os.path.join(img_path,file)
    img0 = cv2.imread(path)  # BGR
    # Padded resize
    img = letterbox(img0, new_shape=imgsz, auto=False,auto_size=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()

    # print(img.shape)
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=True)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                if conf < prob_thres:
                    continue
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
    

    # Print time (inference + NMS)
    print('%sDone. (%.3fs)' % (file, t2 - t1))

    if not os.path.exists("./test_res/"):
        os.makedirs("./test_res/")

    save_path = os.path.join("./test_res/",file)

    cv2.imwrite(save_path, img0)























