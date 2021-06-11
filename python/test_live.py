#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys
import random
import cv2
import numpy as np
import time
from camera import JetCamera
import traceback
import TRTYolov5 as lindsay
import torch
from tools import plot_one_box,LoadStreams,LoadImages,judge_rtsp,self_main

model = torch.load('weights/yolov5x6.pt')['model'].float()  # load to FP32
categories = model.module.names if hasattr(model, 'module') else model.names
globalColors = [[random.randint(0, 255) for _ in range(3)] for _ in categories]
def official_infer_main(source):
    source_flag = judge_rtsp(source, 'rtsp://admin')
    if source_flag:
        print('loading RTSP ...')
        dataset = LoadStreams(source, img_size=640)
    else:
        print('loading Videos ...')
        dataset = LoadImages(source, img_size=640)
    for path, img, im0s, vid_cap in dataset:
        ret = lindsay.detect(engine, im0s, 0.4)
        res = eval(ret)
        if res['num_det'] > 0:
            for ret in res['objects']:
                cls_id, x1, y1, x2, y2 = ret
                plot_one_box(x1, y1, x2, y2, im0s, categories[int(cls_id)], line_thickness=3,
                             color=globalColors[int(cls_id)])
        cv2.imshow(str(path), im0s)
        cv2.waitKey(1)

if __name__ == '__main__':
    engine_path='../yolov5s.engine'
    engine = lindsay.create(engine_path)
    # engine = lindsay.create('../yolov5s.engine')
    flag = 'self_main'  # 'self_main'  自己写的接口,     'official_images'   添加了官方的加载接口
    source='/home/lindsay/Desktop/yolov5_tensorrtx_python-master/videos/trim.avi'
    if flag == 'self_main':
        self_main(source,engine_path, categories, globalColors)
    elif flag == 'official_images':
        official_infer_main(source)



