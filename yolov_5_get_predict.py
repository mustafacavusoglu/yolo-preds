import torch
import os

# Model
model = torch.hub.load('.', 'custom', 'best.pt', source='local')


for root, _, files in os.walk(r'D:\yolo\yolov5\data\dmr\test\\'):
    for file in files:
        if file.endswith('.png'):
            results = model(root + '\\' + file)
            results.save(save_dir = 'preds/img')  # or .show(), .save(), .crop(), .pandas(), etc.