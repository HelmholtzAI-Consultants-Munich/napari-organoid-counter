import numpy as np
from skimage.transform import rescale
from skimage.color import gray2rgb

import torch
import torch.nn as nn

from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from matplotlib import pyplot as plt

def prepare_img(test_img, step, window_size, rescale_factor, trans, device):

    test_img = np.squeeze(test_img)
    test_img = rescale(test_img, rescale_factor, preserve_range=True)
    img_height, img_width = test_img.shape
    pad_x = (img_height//step)*step + window_size - img_height
    pad_y = (img_width//step)*step + window_size - img_width
    test_img = np.pad(test_img, ((0, int(pad_x)), (0, int(pad_y))), mode='edge')

    # The created dataset is also 0-255 jpeg images
    test_img = (test_img-np.min(test_img))/(np.max(test_img)-np.min(test_img)) 
    test_img = (255*test_img).astype(np.uint8)
    test_img = gray2rgb(test_img) #[H,W,C]

    # convert to tensor
    test_img = trans(test_img)
    test_img = torch.unsqueeze(test_img, axis=0) #[B, C, H, W]
    test_img = test_img.to(device)
    
    return test_img

def apply_nms(bbox_preds, scores_preds, iou_thresh=0.5):
    """Function applies non max suppression to iteratively remove lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring) box.

    Args:
        orig_prediction (Dictionary): Model output for each input image.
        iou_thresh (float, optional): IOU threshold. Defaults to 0.3.

    Returns:
        final_prediction: Dictionary containing boxes, scores and labels after nms.
    """
    
    # torchvision returns the indices of the bboxes to keep
    keep = nms(bbox_preds, scores_preds, iou_thresh)
    bbox_preds_kept = bbox_preds[keep]
    scores_preds = scores_preds[keep]
    return bbox_preds_kept, scores_preds

def convert_boxes_to_napari_view(pred_bboxes):
    # convert way boxes are stored so they are correctly represented in napari
    if pred_bboxes is None: return []
    new_boxes = []
    for idx in range(pred_bboxes.size(0)):
        x1_real, y1_real, x2_real, y2_real = pred_bboxes[idx].numpy()
        new_boxes.append(np.array([[x1_real, y1_real],
                                [x1_real, y2_real],
                                [x2_real, y2_real],
                                [x2_real, y1_real]]))
    return new_boxes

def apply_normalization(img):
    img = np.squeeze(img) #self.viewer.layers[self.image_layer_name].data)
    img = img.astype(np.float64)
    #Normalise and return img to range 0-255
    img_min = np.min(img) # 31.3125 png 0
    img_max = np.max(img) # 2899.25 png 178
    img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    return img_norm

class frcnn(nn.Module):
    def __init__(self, num_classes,rpn_score_thresh=0,box_score_thresh=0.05):
        """
        A FRCNN module performs below operations:
        - Loads the pretrained FasterRCNN model.
      """
        super(frcnn, self).__init__()
        self.num_classes = num_classes
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_score_thresh = rpn_score_thresh, box_score_thresh = box_score_thresh)
        #self.model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, rpn_score_thresh = rpn_score_thresh, box_score_thresh = box_score_thresh)        
        # get number of input features for the classifier
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
        
    def forward(self, x, return_all=False):
        self.model.eval()
        return self.model(x)
    
'''
def get_iou(a, b, epsilon=1e-5, intersection_check=False):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width =  (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        if intersection_check:
            return 0.0, False
        else:
            return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    if intersection_check:
        return iou, bool(area_overlap)
    else:
        return iou
'''


'''
def add_text_to_img(img, organoid_number, downsampling=1):
    """
    Adds the number of organoids detected as text to the image and returns it
    Parameters
    ----------
    img: numpy array
        The image on which text needs to be added
    organoid_number: int
        The number of organoids detected - to be added as text 
    downsampling: int
        the downsampling of the image will affect the text size
    Returns
    -------
    img: numpy array
        The image with text added to it
    """
    # define thickness and font size of the text depending on the downsampling rate
    thickness=2
    if downsampling==1: 
        fontSize = 10 #6
        thickness = 12 #4
    elif downsampling<4: fontSize = 3
    else: fontSize = 2
    # add text to image
    img = cv2.putText(img, 
        'Organoids: '+str(organoid_number), 
        org=(round(img.shape[1]*0.05), round(img.shape[0]*0.1)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontSize,
        thickness=thickness,
        color=(255))
    return img
'''


'''
    def is_bbox_there(self, pred_bboxes, cur_bbox, thresh = 0.5):
        is_there = False
        for bbox in pred_bboxes:
            iou = get_iou(bbox, cur_bbox)
            if iou > thresh: 
                is_there=True
                break
        return is_there

    def _pad_and_crop(self, img, i, j, height, width, window_size):
        # crop
        if (i+window_size) < height and (j+window_size) < width:
            img_crop = img[i:(i+window_size), j:(j+window_size), :]
        elif (i+window_size) >= height and (j+window_size) < width:
            pad_size = (i+window_size) - height
            img_crop = img[i:, j:(j+window_size), :]
            img_crop = np.pad(img_crop, ((0, pad_size), (0,0), (0,0)))
        elif (j+window_size) >= width and (i+window_size) < height:
            pad_size = (j+window_size) - width
            img_crop = img[i:(i+window_size), j:, :]
            img_crop = np.pad(img_crop, ((0,0), (0, pad_size), (0,0)))
        else:
            pad_size_i = (i+window_size) - height
            pad_size_j = (j+window_size) - width
            img_crop = img[i:, j:, :]
            img_crop = np.pad(img_crop, ((0, pad_size_i), (0, pad_size_j), (0,0)))
        return img_crop

    def run_sliding_window(self, 
                           img, 
                           height, 
                           width, 
                           step, 
                           window_size, 
                           rescale_factor, 
                           confidence):
        pred_bboxes = []
        # loop through patches
        for i in range(0, height, step):
            for j in range(0, width, step):
                print(i,j)
                img_crop = self._pad_and_crop(img, i, j, height, width, window_size)
                img_crop = (img_crop-np.min(img_crop))/(np.max(img_crop)-np.min(img_crop))
                img_crop = (255*img_crop).astype(np.uint8)

                # convert to tensor
                img_crop = self.transfroms(img_crop)
                img_crop = torch.unsqueeze(img_crop, axis=0)
                img_crop = img_crop.to(self.device)
                
                # get predictions
                output = self.model(img_crop.float())
                preds = output[0]['boxes'].cpu().detach()
                scores = output[0]['scores'].cpu().detach().numpy()
                if preds.size(0)==0: continue
                else:
                    for bbox_id in range(preds.size(0)):
                        bbox = preds[bbox_id]
                        score = scores[bbox_id]
                        print(score)
                        if score > confidence:
                            print('here')
                            y1, x1, y2, x2 = bbox # predictions from model will be in form x1,y1,x2,y2
                            x1_real = (x1.item() + i) // rescale_factor
                            x2_real = (x2.item() + i) // rescale_factor
                            y1_real = (y1.item() + j) // rescale_factor
                            y2_real = (y2.item() + j) // rescale_factor
                            if not self.is_bbox_there(pred_bboxes, (x1_real, y1_real, x2_real, y2_real)):
                                pred_bboxes.append([x1_real, y1_real, x2_real, y2_real])
        return pred_bboxes

    def run(self, 
            img, 
            window_sizes,
            downsampling, 
            min_diameter, 
            confidence,    
            window_overlap):
        

        # resize image and convert to rgb as network expects
        rescale_factor = 1 / downsampling # default = 0.5
        step = round(window_size * window_overlap * rescale_factor)
        img = rescale(img, rescale_factor, preserve_range=True)
        img = gray2rgb(img) #img is HxWxC

        # run sliding window
        pred_bboxes = self.run_sliding_window(img, img.shape[0], img.shape[1], step, window_size, rescale_factor, confidence)

        # convert way boxes are stored so they are correctly represented in napari
        for idx in range(len(pred_bboxes)):
            x1_real, y1_real, x2_real, y2_real = pred_bboxes[idx]
            pred_bboxes[idx] = np.array([[x1_real, y1_real],
                                        [x1_real, y2_real],
                                        [x2_real, y2_real],
                                        [x2_real, y1_real]])
        return pred_bboxes
'''