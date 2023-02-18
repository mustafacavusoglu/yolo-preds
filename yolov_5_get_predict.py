import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

model = torch.hub.load('.', 'custom', r'best.pt', source='local')

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def reverse_convert(xcenter, ycenter, w, h, w_img, h_img):
    h = h * h_img
    w = w * w_img
    x = (xcenter * w_img) - w/2
    y = (ycenter * h_img) - h/2
    return x, y, w + x,  y + h


ext = '.png'

for root, _, files in os.walk(r'data/cbis_cc/test/'):
    for file in files:
        if file.endswith(ext):
            image = cv2.imread(root + '/' + file)
            image2 = image.copy()

            if os.path.exists(root + '/' + file.replace(ext, '.txt')):
                lines = []
                with open(root + '/' + file.replace(ext, '.txt'), 'r') as f:
                    line = f.readlines()
                    for i in line:
                        lines.append(i.split(' '))

                n_lines = []    
                for i in lines:
                    x1, y1, x2, y2 = reverse_convert(float(i[1]), float(i[2]), float(i[3]), float(i[4]), 1024, 1024)
                    n_lines.append([x1, y1, x2, y2])
                
                for n in n_lines:
                    n = [int(i) for i in n]
                    cv2.rectangle(image, (n[0], n[1]), (n[2], n[3]), (0, 255, 0), 2)

            results = model(root + '/' + file)
            coors = results.pandas().xyxy[0]
            
            if len(coors.xmin):
                ious = []
                for i in range(len(coors)):
                    xmin = int(coors.xmin[i])
                    xmax = int(coors.xmax[i])
                    ymin = int(coors.ymin[i])
                    ymax = int(coors.ymax[i])

                    cv2.rectangle(image, (xmin,  ymin), (xmax, ymax), (0, 0, 255), 2)

                    if os.path.exists(root + '/' + file.replace(ext, '.txt')):
                        for n in n_lines:
                            iou = bb_intersection_over_union(n, [coors.xmin[i], coors.ymin[i], coors.xmax[i], coors.ymax[i]])
                            ious.append(iou)
                        iouu = sum(ious) / len(n_lines)
                    else:
                        iouu = 0
                if iouu > 0:
                    cv2.imwrite('path_to_save/' + file.replace(ext, f'_iou_{iouu:.2f}_.png'), image)
                else:
                    cv2.imwrite('path_to_save/' + file.replace(ext, f'_no_iou_.png'), image)
            else:
                cv2.imwrite('path_to_save/' + file.replace(ext, f'_no_preds_.png'), image)

