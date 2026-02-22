from contextlib import contextmanager
import os
from pathlib import Path

import numpy as np
import pandas as pd
import math
import json
import csv
from skimage.transform import rescale, resize
from skimage.color import gray2rgb

import torch
from torchvision.ops import nms

from napari_organoid_counter import settings
import torch.nn.functional as F

EXCLUDED_MODELS = [
    "ssd_organoid_best_coco_bbox_mAP_epoch_86.onnx",
    "rtmdet_l_organoid_best_coco_bbox_mAP_epoch_323.onnx",
]

def add_local_models():
    """ Checks the models directory for any local models previously added by the user.
    If some are found then these are added to the model dictionary (see settings). """
    if not os.path.exists(settings.MODELS_DIR): return
    model_names_in_dir = [file for file in os.listdir(settings.MODELS_DIR)]
    model_names_in_dict = [settings.MODELS[key]["filename"] for key in settings.MODELS.keys()]
    for model_name in model_names_in_dir:
        if model_name not in model_names_in_dict and model_name.endswith(settings.MODEL_TYPE) and model_name not in EXCLUDED_MODELS:
            _ = add_to_dict(model_name)

def add_to_dict(filepath):
    """ Given the full path and name of a model in filepath the model is added to the models dict (see settings)"""
    filepath = Path(filepath)
    name = filepath.name
    stem_name = filepath.stem
    settings.MODELS[stem_name] = {"filename": name, "source": "local"}
    return stem_name

def return_is_file(path, filename):
    """ Return True if the file exists in path and False otherwise """
    full_path = join_paths(path, filename)
    return os.path.isfile(full_path)

def join_paths(path1, path2):
    """ Returns output of os.path.join """
    return os.path.join(path1, path2)

@contextmanager
def set_dict_key(dictionary, key, value):
    """ Used to set a new value in the napari layer metadata """
    dictionary[key] = value
    yield
    del dictionary[key]

def get_diams(bbox):
    """ Get the lengths of the bounding boxes """
    x1_real, y1_real, x2_real, y2_real = bbox
    dx = abs(x1_real - x2_real)
    dy = abs(y1_real - y2_real)
    return dx, dy

def write_to_json(name, data):
    """ Write data to a json file. Here data is a dict """
    with open(name, 'w') as outfile:
        json.dump(data, outfile)  

def get_bboxes_as_dict(bboxes, bbox_ids, scores, scales, labels):
    """ Write all data, boxes, ids and scores, scale and class label, to a dict so we can later save as a json """
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
                                                'scale_y': str(scales[1]),
                                                'label': labels[idx],
                                                }
                        })
    return data_json

def write_to_csv(name, data):
    """ Write data to a csv file. Here data is a list of lists, where each item represents a row in the csv file. """
    df = pd.DataFrame(data, columns=['OrganoidID', 'D1[um]', 'D2[um]', 'Area[um^2]', 'Label'])
    df.to_csv(name, index=False, sep=';')

def get_bbox_diameters(bboxes, bbox_ids, scales, labels):
    """ Write all data, box diameters and area, ids, scale and labels to a list so we can later save as a csv """
    data_csv = []
    # save diameters and area of organoids (approximated as ellipses)
    for idx, bbox, label in zip(bbox_ids, bboxes, labels):
        d1 = abs(bbox[0][0] - bbox[2][0]) * scales[0]  # X direction (width)
        d2 = abs(bbox[0][1] - bbox[2][1]) * scales[1]  # Y direction (height)
        area = math.pi * d1 * d2 / 4  # divide by 4 because d1 and d2 are full diameters, not semi-axes
        data_csv.append([idx, round(d1,3), round(d2,3), round(area,3), label])
    return data_csv

def squeeze_img(img):
    """ Squeeze image - all dims that have size one will be removed """
    return np.squeeze(img)

def resize_keep_ratio_numpy(img, scale=(416, 416)):
    """
    Resize a numpy array [B, C, H, W] keeping aspect ratio using skimage.

    Args:
        img (np.ndarray): shape [B, C, H, W]
        scale (tuple): target maximum (w, h), e.g. (416, 416)

    Returns:
        np.ndarray: resized array [B, C, new_h, new_w]
        (int, int): new size (new_h, new_w)
        float: scale factor
    """
    assert img.ndim == 4, "Expected input shape [B, C, H, W]"
    B, C, H, W = img.shape
    target_w, target_h = scale

    # compute scale factor
    scale_factor = min(target_w / W, target_h / H)
    new_w, new_h = int(W * scale_factor), int(H * scale_factor)

    resized_batch = []
    for i in range(B):
        # [C, H, W] → [H, W, C]
        img_hwc = np.transpose(img[i], (1, 2, 0))
        resized = resize(img_hwc, (new_h, new_w), order=1, anti_aliasing=True)
        # back to [C, H, W]
        resized_chw = np.transpose(resized, (2, 0, 1))
        resized_batch.append(resized_chw.astype(np.float32))

    resized_batch = np.stack(resized_batch, axis=0)
    return resized_batch 

def resize_keep_ratio_torch(img, scale=(416, 416)):
    """
    Resize a numpy array [B, C, H, W] keeping aspect ratio using PyTorch.
    
    Args:
        img (np.ndarray): shape [B, C, H, W]
        scale (tuple): target maximum (w, h), e.g. (416, 416)
    Returns:
        torch.Tensor: resized array [B, C, new_h, new_w]
    """
    assert img.ndim == 4, "Expected input shape [B, C, H, W]"
    B, C, H, W = img.shape
    target_w, target_h = scale
    
    # Compute scale factor
    scale_factor = min(target_w / W, target_h / H)
    new_w, new_h = int(W * scale_factor), int(H * scale_factor)
    
    # Resize using bilinear interpolation (vectorized for entire batch)
    resized_tensor = F.interpolate(
        img,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False,
        antialias=True
    )
    
    # Convert back to numpy
    resized_batch = resized_tensor.cpu().numpy().astype(np.float32)
    
    return resized_batch

def prepare_img_onnx(test_img, step, window_size, rescale_factor):
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

    # convert from RGB to GBR to match exported model preprocessing
    test_img = test_img[..., ::-1] 

    test_img = test_img.astype(np.float32) / 255.0
    # HWC -> CHW
    test_img = np.transpose(test_img, (2, 0, 1))
    # Add batch dimension
    test_img = np.expand_dims(test_img, axis=0)
    test_img = torch.from_numpy(test_img).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return test_img, img_height, img_width

def apply_nms(bbox_preds, scores_preds, labels_preds, iou_thresh=0.5):
    """ Function applies non max suppression to iteratively remove lower scoring boxes which have an IoU greater than iou_threshold 
    with another (higher scoring) box. The boxes and corresponding scores whihc remain are returned. """
    # torchvision returns the indices of the bboxes to keep
    keep = nms(bbox_preds, scores_preds, iou_thresh)
    # filter existing boxes and scores and return
    bbox_preds_kept = bbox_preds[keep]
    scores_preds = scores_preds[keep]
    labels_preds = labels_preds[keep]
    return bbox_preds_kept, scores_preds, labels_preds

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
    if len(new_boxes) > 0: new_boxes = torch.stack(new_boxes)
    return new_boxes

def apply_normalization(img):
    """ Normalize image"""
    # squeeze and change dtype
    img = squeeze_img(img)
    img = img.astype(np.float64)
    # adapt img to range 0-255
    img_min = np.min(img) # 31.3125 png 0
    img_max = np.max(img) # 2899.25 png 178
    if img_min < 0 or img_max > 255:
        img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    else:
        img_norm = img.astype(np.uint8)
    return img_norm

def get_edge_color(labels, use_default_color: bool):
    edge_color = []
    if use_default_color:  # Detection-Only mode or Deterction only model
        edge_color = [settings.COLOR_DEFAULT] * len(labels)  # Set all edges to default color (magenta)
    else:  # For other annotation modes (Binary Classification, 3 classes, etc.)
        for label in labels:
            if int(label) == -1:  # Uncertain labels in Binary Classification Mode
                edge_color.append(settings.COLOR_DEFAULT)  # Set edge color to default for uncertain labels
            else:
                edge_color.append(settings.COLOR_MAPPING[int(label)][0])  # Set edge color based on the predicted label
    return edge_color
