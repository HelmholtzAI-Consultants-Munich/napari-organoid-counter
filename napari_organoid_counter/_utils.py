from contextlib import contextmanager
import numpy as np
import math
import json
import csv
from skimage.transform import rescale
from skimage.color import gray2rgb

import torch
import torch.nn as nn
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

@contextmanager
def set_dict_key(dictionary, key, value):
    """ Used to set a new value in the napari layer metadata """
    dictionary[key] = value
    yield
    del dictionary[key]

def write_to_json(name, data):
    """ Write data to a json file. Here data is a dict """
    with open(name, 'w') as outfile:
        json.dump(data, outfile)  

def get_bboxes_as_dict(bboxes, bbox_ids, scores, scales):
    """ Write all data, boxes, ids and scores and scale, to a dict so we can later save as a json """
    data_json = {} 
    for idx, bbox in enumerate(bboxes):
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        data_json.update({str(bbox_ids[idx]): {'box_id': str(bbox_ids[idx]),
                                                'x1': str(x1),
                                                'x2': str(x2),
                                                'y1': str(y1),
                                                'y2': str(y2),
                                                'confidence': str(scores[idx]),
                                                'scale_x': str(scales[0]),
                                                'scale_y': str(scales[1])
                                                }
                        })
    return data_json

def write_to_csv(name, data):
    """ Write data to a csv file. Here data is a list of lists, where each item represents a row in the csv file. """
    with open(name, 'w') as f:
        write = csv.writer(f, delimiter=';')
        write.writerow(['OrganoidID', 'D1[um]','D2[um]', 'Area [um^2]'])
        write.writerows(data)

def get_bbox_diameters(bboxes, bbox_ids, scales):
    """ Write all data, box diameters and area, ids and scale, to a list so we can later save as a csv """
    data_csv = []
    # save diameters and area of organoids (approximated as ellipses)
    for idx, bbox in enumerate(bboxes):
        d1 = abs(bbox[0][0] - bbox[2][0]) * scales[0]
        d2 = abs(bbox[0][1] - bbox[2][1]) * scales[1]
        area = math.pi * d1 * d2
        data_csv.append([bbox_ids[idx], round(d1,3), round(d2,3), round(area,3)])
    return data_csv

def squeeze_img(img):
    """ Squeeze image - all dims that have size one will be removed """
    return np.squeeze(img)

def prepare_img(test_img, step, window_size, rescale_factor, trans, device):
    """ The original image is prepared for running model inference """
    # squeeze and resize image
    test_img = squeeze_img(test_img)
    test_img = rescale(test_img, rescale_factor, preserve_range=True)
    img_height, img_width = test_img.shape
    # pad image
    pad_x = (img_height//step)*step + window_size - img_height
    pad_y = (img_width//step)*step + window_size - img_width
    test_img = np.pad(test_img, ((0, int(pad_x)), (0, int(pad_y))), mode='edge')

    # normalise and convert to RGB - model input has size 3
    test_img = (test_img-np.min(test_img))/(np.max(test_img)-np.min(test_img)) 
    test_img = (255*test_img).astype(np.uint8)
    test_img = gray2rgb(test_img) #[H,W,C]

    # convert to tensor and send to device
    test_img = trans(test_img)
    test_img = torch.unsqueeze(test_img, axis=0) #[B, C, H, W]
    test_img = test_img.to(device)
    
    return test_img

def apply_nms(bbox_preds, scores_preds, iou_thresh=0.5):
    """ Function applies non max suppression to iteratively remove lower scoring boxes which have an IoU greater than iou_threshold 
    with another (higher scoring) box. The boxes and corresponding scores whihc remain are returned. """
    # torchvision returns the indices of the bboxes to keep
    keep = nms(bbox_preds, scores_preds, iou_thresh)
    # filter existing boxes and scores and return
    bbox_preds_kept = bbox_preds[keep]
    scores_preds = scores_preds[keep]
    return bbox_preds_kept, scores_preds

def convert_boxes_to_napari_view(pred_bboxes):
    """ The bboxes are converted from tensors in model output form to a form which can be visualised in the napari viewer """
    if pred_bboxes is None: return []
    new_boxes = []
    for idx in range(pred_bboxes.size(0)):
        # convert to numpy and take coordinates 
        x1_real, y1_real, x2_real, y2_real = pred_bboxes[idx].numpy()
        # append to a list in form napari exects
        new_boxes.append(np.array([[x1_real, y1_real],
                                [x1_real, y2_real],
                                [x2_real, y2_real],
                                [x2_real, y1_real]]))
    return new_boxes

def convert_boxes_from_napari_view(pred_bboxes):
    """ The bboxes are converted from the form they were in the napari viewer to tensors that correspond to the model output form """
    new_boxes = []
    for idx in range(len(pred_bboxes)):
        # read coordinates
        x1 = pred_bboxes[idx][0][0]
        x2 = pred_bboxes[idx][2][0]
        y1 = pred_bboxes[idx][0][1]
        y2 = pred_bboxes[idx][2][1]
        # convert to tensor and append to list
        new_boxes.append(torch.Tensor([x1, y1, x2, y2]))
    new_boxes = torch.stack(new_boxes)
    return new_boxes

def apply_normalization(img):
    """ Normalize image"""
    # squeeze and change dtype
    img = squeeze_img(img)
    img = img.astype(np.float64)
    # adapt img to range 0-255
    img_min = np.min(img) # 31.3125 png 0
    img_max = np.max(img) # 2899.25 png 178
    img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    return img_norm

class frcnn(nn.Module):
    def __init__(self, num_classes,rpn_score_thresh=0,box_score_thresh=0.05):
        """ An FRCNN module loads the pretrained FasterRCNN model """
        super(frcnn, self).__init__()
        # define classes and load pretrained model
        self.num_classes = num_classes
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_score_thresh = rpn_score_thresh, box_score_thresh = box_score_thresh)
        # get number of input features for the classifier
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
        self.model.eval()

    def forward(self, x, return_all=False):
        """ A forward pass through the model """
        return self.model(x)
 