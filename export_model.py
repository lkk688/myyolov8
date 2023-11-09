import torch
from ultralytics import YOLO
import cv2
from ultralytics.data.augment import LetterBox
import numpy as np
from ultralytics.utils import ops

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def export_statedicts():
        # Load a model
    model = YOLO('yolov8x.yaml')  # build a new model from scratch
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training) _load in ultralytics\engine\model.py

    detectionmodel=model.model #model.ckpt.model, 
    torch.save(detectionmodel.state_dict(), './yolov8x_statedicts.pt')

if __name__ == "__main__":
    model = YOLO('yolov8x.yaml')  # build a new model from scratch
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training) _load in ultralytics\engine\model.py

    #call export in ultralytics\engine\model.py, then class Exporter in ultralytics\engine\exporter.py
    format='engine'#'onnx' 
    results = model.export(format=format)  # export the model to ONNX format