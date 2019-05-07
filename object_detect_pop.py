from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime

from PIL import Image

import torch
from torch.autograd import Variable


# function that gets the img_path, and outputs the Yolov3 logits


def Yolov3(img_paths, model_def="config/yolov3.cfg", img_size = 416, weights_path="weights/yolov3.weights",
           class_path = "data/coco.names", debug=False, conf_thres=0.5, nms_thres=0.4, addr_to_Yolov3=''):
    """

    :param img_paths: "path to image dataset"
    :param model_def: "path to model definition file"
    :param img_size: "size of each image dimension, 416 or 608"
    :param weights_path: "path to weights file"
    :param class_path: "path to class label file"
    :param debug: debug flag
    :param conf_thres: "object confidence threshold"
    :param nms_thres: "iou thresshold for non-maximum suppression"
    :return:
    """
    model_def = addr_to_Yolov3+model_def
    weights_path = addr_to_Yolov3 + weights_path
    class_path = addr_to_Yolov3 + class_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    img_list=[]

    for img_path in img_paths:
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, img_size)
        img_list.append(img)

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    if(debug):
        print("\nPerforming object detection:")
        prev_time = time.time()

    for input_imgs in img_list:
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs.unsqueeze(0))
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Log progress
        # (x1, y1, x2, y2, object_conf, class_score, class_pred)
        if (debug):
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+Inference Time: %s" % (inference_time))

        # Save detections
        img_detections.extend(detections)


    # Iterate through images and save plot of detections

    results=[]
    for img_i in range(len(img_detections)):

        detections=img_detections[img_i]
        path=img_paths[img_i]

        if (debug):
            print("(%d) Image: '%s'" % (img_i, path))


        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            result_in_an_image=[]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                
                if(debug):
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                result_in_an_image.append([x1.item(),y2.item(),x2.item(),y2.item(), conf.item(),cls_conf.item(),classes[int(cls_pred)]])

            results.append(result_in_an_image)

    return results
