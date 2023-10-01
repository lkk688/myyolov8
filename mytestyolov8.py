import torch
from ultralytics import YOLO
import cv2
from ultralytics.data.augment import LetterBox
import numpy as np
from ultralytics.utils import ops


def test_inference(detectionmodel, imagepath):   
    nn_module = isinstance(detectionmodel, torch.nn.Module)
    device = 'cuda:0'
    fuse = False
    fp16 = False #model.fp16
    if nn_module:  # in-memory PyTorch model
        detectionmodel = detectionmodel.to(device)
        detectionmodel = detectionmodel.fuse(verbose=False) if fuse else detectionmodel
        #stride = max(int(detectionmodel.stride.max()), 32)  # model stride
        names = detectionmodel.module.names if hasattr(detectionmodel, 'module') else detectionmodel.names  # get class names
        detectionmodel.half() if fp16 else detectionmodel.float()
        detectionmodel.eval()

        im0 = cv2.imread(imagepath) #(1080, 810, 3)
        im0s = [im0] #image list
        #preprocess(im0s)
        letterbox = LetterBox((640, 640), auto=True, stride=32)
        processedimgs=[letterbox(image=x) for x in im0s] #list of (640, 480, 3)
        im = np.stack(processedimgs) #(1, 640, 480, 3)
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        im = im.to(device)
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        im /= 255
        #inference
        preds = detectionmodel(im) 
        #postprocess in ultralytics\models\yolo\detect\predict.py 
        #input preds: (batch_size, num_classes + 4 + num_masks, num_boxes)
        preds = ops.non_max_suppression(preds, conf_thres=0.25, iou_thres= 0.45, agnostic=False, classes=None)
        #output: Returns:
            # (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            #     shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            #     (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        result=[] #List[Dict[str, Tensor]]
        resdict={}
        for i, pred in enumerate(preds):
            orig_img = im0s[i]
            pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], orig_img.shape)
            resdict["boxes"] = pred[:, :4].detach().cpu().numpy()
            resdict["scores"] = pred[:, 4].detach().cpu().numpy()
            resdict["labels"] = pred[:, 5].detach().cpu().numpy() #class
            result.append(resdict)
            #result[i]["boxes"] = resdict
        print(result) #list of one [6, 6]
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__":

    # Load a model
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training) _load in ultralytics\engine\model.py

    imagepath = 'https://ultralytics.com/images/bus.jpg'
    #result = test_inference(model, imagepath)

    #results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    #return list of Results object, key 'boxes'
    # Use the model
    results = model.train(data='coco128.yaml', epochs=3)  # train the model
    results = model.val()  # evaluate model performance on the validation set

    # results = model.export(format='onnx')  # export the model to ONNX format

    # detectionmodel=model.model #model.ckpt.model, 
    # torch.save(detectionmodel.state_dict(), './yolov8n_statedicts.pt')


    # imagepath=r'C:\Users\lkk68\Documents\GitHub\myyolov8\bus.jpg'
    # test_inference(detectionmodel, imagepath)
