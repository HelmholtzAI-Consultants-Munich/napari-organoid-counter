
import torch
from torchvision.transforms import ToTensor
from napari_organoid_counter._utils import frcnn, prepare_img, apply_nms, convert_boxes_to_napari_view

class OrganoiDL():
    def __init__(self, model_checkpoint='model-weights/model_v1.ckpt'):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = frcnn(num_classes=2, rpn_score_thresh=0, box_score_thresh = 0.05)
        ckpt = torch.load(model_checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt) #.state_dict())
        self.model = self.model.to(self.device)
        self.transfroms = ToTensor()

        self.pred_bboxes = None
        self.pred_scores = None
        self.img_scale = None

    def sliding_window(self, test_img, step, window_size, rescale_factor, pred_bboxes=[], scores_list=[]):
    
        img_height, img_width = test_img.size(2), test_img.size(3)

        for i in range(0, img_height, step):
            for j in range(0, img_width, step):
                # crop
                img_crop = test_img[:, :, i:(i+window_size), j:(j+window_size)]
                # get predictions
                output = self.model(img_crop.float())
                preds = output[0]['boxes']
                if preds.size(0)==0: continue
                else:
                    for bbox_id in range(preds.size(0)):
                        y1, x1, y2, x2 = preds[bbox_id].cpu().detach() # predictions from model will be in form x1,y1,x2,y2
                        x1_real = torch.div(x1+i, rescale_factor, rounding_mode='floor')
                        x2_real = torch.div(x2+i, rescale_factor, rounding_mode='floor')
                        y1_real = torch.div(y1+j, rescale_factor, rounding_mode='floor')
                        y2_real = torch.div(y2+j, rescale_factor, rounding_mode='floor')
                        pred_bboxes.append(torch.Tensor([x1_real, y1_real, x2_real, y2_real]))
                        scores_list.append(output[0]['scores'][bbox_id].cpu().detach())
        return pred_bboxes, scores_list

    def run(self, 
            img, 
            img_scale,
            window_sizes,
            downsampling_sizes,   
            window_overlap):
        
        # run for all window sizes
        bboxes = []
        scores = []

        self.img_scale = img_scale

        for window_size, downsampling in zip(window_sizes, downsampling_sizes):
            # compute the step for the sliding window, based on window overlap
            rescale_factor = 1 / downsampling
            # window size after rescaling
            window_size = round(window_size * rescale_factor)
            step = round(window_size * window_overlap)
            # prepare image for model - norm, tensor, etc.
            ready_img = prepare_img(img, step, window_size, rescale_factor, self.transfroms , self.device)
            bboxes, scores = self.sliding_window(ready_img, step, window_size, rescale_factor, bboxes, scores)

        bboxes = torch.stack(bboxes)
        scores = torch.stack(scores)
        # apply NMS to remove overlaping boxes
        self.pred_bboxes, self.pred_scores = apply_nms(bboxes, scores)

    def apply_params(self, confidence, min_diameter_um):
        pred_bboxes = self.apply_confidence_thresh(confidence)
        pred_bboxes = self.filter_small_organoids(min_diameter_um, pred_bboxes)
        pred_bboxes = convert_boxes_to_napari_view(pred_bboxes)
        return pred_bboxes

    def apply_confidence_thresh(self, confidence):
        if self.pred_bboxes is None: return None
        keep = (self.pred_scores>confidence).nonzero(as_tuple=True)[0]
        result_bboxes = self.pred_bboxes[keep]
        return result_bboxes

    def filter_small_organoids(self, min_diameter_um, pred_bboxes):
        if pred_bboxes is None: return None
        if len(pred_bboxes)==0: return None
        min_diameter_x = min_diameter_um / self.img_scale[0]
        min_diameter_y = min_diameter_um / self.img_scale[1]
        keep = []
        for idx in range(len(pred_bboxes)):
            x1_real, y1_real, x2_real, y2_real = pred_bboxes[idx]
            dx = abs(x1_real - x2_real)
            dy = abs(y1_real - y2_real)
            if dx >= min_diameter_x and dy >= min_diameter_y: keep.append(idx) 
        pred_bboxes = pred_bboxes[keep]
        return pred_bboxes


