import cv2
import time
import random
import numpy as np
import onnxruntime as ort
import os
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

cuda = True
model = "best.onnx"


providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(model, providers=providers)


def letterbox(im, new_shape=(448, 448), color=(114, 114, 114), 
            auto=True, scaleup=True, stride=32):
    
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        ratio = min(ratio, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    image_ = im.transpose((2, 0, 1))
    image_ = np.expand_dims(image_, 0)
    image_ = np.ascontiguousarray(image_)

    im = image_.astype(np.float32)
    im /= 255

    outname = [i.name for i in session.get_outputs()]

    inname = [i.name for i in session.get_inputs()]

    inp = {inname[0]:im}
    outputs = session.run(outname, inp)[0]
    return im, outputs, ratio, (dw, dh)

names = ['lesion']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
#%%
# TODO for loop over img_files
prediction_path = r'D:\yolo\yolov7\data\test\\'
image_files = []
for root, _, files in os.walk(prediction_path):
    if len(files) > 2:
        for file in files:
            if file.endswith('.png'):
                image_files.append(root + '\\' + file)




for img_name in image_files:
    print(img_name + 'PREDİCTİNG...')
    img = cv2.imread(img_name)
    #img = cv2.resize(img, (320, 320), interpolation= cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    image = img.copy()
    image_,outputs, ratio, dwdh = letterbox(image, auto=False)
    print(str(outputs.shape[0]) + ' LESION DETECTED')
    ori_images = [img.copy()]


    if outputs.shape[0] > 0:
        # with open(img_name[:-4] + '_prediction.txt', 'w') as f:
        for batch_id, x0, y0, x1, y1, cls_id, score in outputs:
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            # if score > 0.85:
            name = names[cls_id]
            color = (0, 255, 180) #colors[name]
            name += f' {str(score)}'
            
            # f.writelines(str(box) + '\n')
            cv2.rectangle(image,box[:2],box[2:],color,1)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
            name = img_name.split('\\')
            name = name[-1][:-4] + '.png'
            cv2.imwrite('preds/' + name, image)
            print(name + ' predicted and saved')
    else:
        name = img_name.split('\\')
        name = name[-1][:-4] + '.png'
        cv2.imwrite('preds/' + name, image)
        print(name + ' predicted and saved')
            
            

""" cv2.imshow("Detected",image)
cv2.waitKey(0)
cv2.destroyAllWindows() """



