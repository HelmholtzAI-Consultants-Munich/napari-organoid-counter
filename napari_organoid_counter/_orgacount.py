import numpy as np
from skimage.transform import rescale
from skimage.color import gray2rgb
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True,rpn_score_thresh = rpn_score_thresh,box_score_thresh = box_score_thresh)        
        # get number of input features for the classifier
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
        
    def forward(self, x, return_all=False):
        self.model.eval()
        return self.model(x)


class OrganoiDL():
    def __init__(self,
                window_size = 512,
                window_overlap = 0.5,
                model_checkpoint='model-weights/tst.ckpt'):
        super().__init__()
        
        self.window_size = window_size
        #self.window_overlap = window_overlap
        self.step = round(self.window_size * window_overlap)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = frcnn(num_classes=2, rpn_score_thresh=0.9, box_score_thresh = 0.85)
        ckpt = torch.load(model_checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt) #.state_dict())
        self.model = self.model.to(self.device)
        self.transfroms = ToTensor()

    def is_bbox_there(self, pred_bboxes, cur_bbox, thresh = 0.5):
        is_there = False
        for bbox in pred_bboxes:
            iou = get_iou(bbox, cur_bbox)
            if iou > thresh: 
                is_there=True
                break
        return is_there

    def run(self, img):
        pred_bboxes = []
        rescale_factor = 0.5
        height, width = img.shape

        for i in range(0, height, self.step):
            for j in range(0, width, self.step):
                
                # crop
                if (i+self.window_size) < height and (j+self.window_size) < width:
                    img_crop = img[i:(i+self.window_size), j:(j+self.window_size)]
                elif (i+self.window_size) >= height and (j+self.window_size) < width:
                    pad_size = (i+self.window_size) - height
                    img_crop = img[i:, j:(j+self.window_size)]
                    img_crop = np.pad(img_crop, ((0, pad_size), (0,0)))
                elif (j+self.window_size) >= width and (i+self.window_size) < height:
                    pad_size = (j+self.window_size) - width
                    img_crop = img[i:(i+self.window_size), j:]
                    img_crop = np.pad(img_crop, ((0,0), (0, pad_size)))
                else:
                    pad_size_i = (i+self.window_size) - height
                    pad_size_j = (j+self.window_size) - width
                    img_crop = img[i:, j:]
                    img_crop = np.pad(img_crop, ((0, pad_size_i), (0, pad_size_j)))
                    
                # convert to tensor
                img_crop = (img_crop-np.min(img_crop))/(np.max(img_crop)-np.min(img_crop))
                img_crop = (255*img_crop).astype(np.uint8)
                img_crop = rescale(img_crop, rescale_factor)
                img_crop = gray2rgb(img_crop)
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
                        y1, x1, y2, x2 = bbox # predictions from model will be in form x1,y1,x2,y2
                        x1_real = (x1/rescale_factor) + i
                        x2_real = (x2/rescale_factor) + i
                        y1_real = (y1/rescale_factor) + j
                        y2_real = (y2/rescale_factor) + j
                        if not self.is_bbox_there(pred_bboxes, (x1_real, y1_real, x2_real, y2_real)):
                            pred_bboxes.append([x1_real, y1_real, x2_real, y2_real])
        return pred_bboxes



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