import os
import torch
from torchvision.transforms import ToTensor
from napari_organoid_counter._utils import frcnn, prepare_img, apply_nms, convert_boxes_to_napari_view, convert_boxes_from_napari_view, get_diams
import subprocess

class OrganoiDL():
    def __init__(self, 
                 img_scale,
                 model_checkpoint=None):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cur_confidence = 0.05
        self.cur_min_diam = 30
        self.model = frcnn(num_classes=2, rpn_score_thresh=0, box_score_thresh = self.cur_confidence)
        self.model_checkpoint = model_checkpoint
        if not os.path.isfile(self.model_checkpoint):
            self.download_model()
        self.load_model_checkpoint(self.model_checkpoint)
        self.model = self.model.to(self.device)
        self.transfroms = ToTensor()

        self.img_scale = img_scale
        self.pred_bboxes = {}
        self.pred_scores = {}
        self.pred_ids = {}
        self.next_id = {}

    def download_model(self):
        subprocess.run(["zenodo_get","10.5281/zenodo.7708763","-o", "model"])
        self.model_checkpoint = os.path.join(os.getcwd(), 'model', 'model_v1.ckpt')
        #zenodo_get(['10.5281/zenodo.7708763'])

    def load_model_checkpoint(self, model_path):
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt) #.state_dict())

    def sliding_window(self, test_img, step, window_size, rescale_factor, prepadded_height, prepadded_width, pred_bboxes=[], scores_list=[]):
    
        #img_height, img_width = test_img.size(2), test_img.size(3)

        for i in range(0, prepadded_height, step):
            for j in range(0, prepadded_width, step):
                
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
            shapes_name,
            window_sizes,
            downsampling_sizes,   
            window_overlap):
        
        # run for all window sizes
        bboxes = []
        scores = []
        
        for window_size, downsampling in zip(window_sizes, downsampling_sizes):
            # compute the step for the sliding window, based on window overlap
            rescale_factor = 1 / downsampling
            # window size after rescaling
            window_size = round(window_size * rescale_factor)
            step = round(window_size * window_overlap)
            # prepare image for model - norm, tensor, etc.
            ready_img, prepadded_height, prepadded_width  = prepare_img(img, step, window_size, rescale_factor, self.transfroms , self.device)
            bboxes, scores = self.sliding_window(ready_img, step, window_size, rescale_factor, prepadded_height, prepadded_width, bboxes, scores)

        bboxes = torch.stack(bboxes)
        scores = torch.stack(scores)
        # apply NMS to remove overlaping boxes
        bboxes, pred_scores = apply_nms(bboxes, scores)
        self.pred_bboxes[shapes_name] = bboxes
        self.pred_scores[shapes_name] = pred_scores
        num_predictions = bboxes.size(0)
        self.pred_ids[shapes_name] = [(i+1) for i in range(num_predictions)]
        self.next_id[shapes_name] = num_predictions+1

    def apply_params(self, shapes_name, confidence, min_diameter_um):
        self.cur_confidence = confidence
        self.cur_min_diam = min_diameter_um
        pred_bboxes, pred_scores, pred_ids = self._apply_confidence_thresh(shapes_name)
        if pred_bboxes.size(0)!=0:
            pred_bboxes, pred_scores, pred_ids = self._filter_small_organoids(pred_bboxes, pred_scores, pred_ids)
        pred_bboxes = convert_boxes_to_napari_view(pred_bboxes)
        return pred_bboxes, pred_scores, pred_ids

    def _apply_confidence_thresh(self, shapes_name):
        if shapes_name not in self.pred_bboxes.keys(): return torch.empty((0))
        keep = (self.pred_scores[shapes_name]>self.cur_confidence).nonzero(as_tuple=True)[0]
        result_bboxes = self.pred_bboxes[shapes_name][keep]
        result_scores = self.pred_scores[shapes_name][keep]
        result_ids = [self.pred_ids[shapes_name][int(i)] for i in keep.tolist()]
        return result_bboxes, result_scores, result_ids
    
    def _filter_small_organoids(self, pred_bboxes, pred_scores, pred_ids):
        if pred_bboxes is None: return None
        if len(pred_bboxes)==0: return None
        min_diameter_x = self.cur_min_diam / self.img_scale[0]
        min_diameter_y = self.cur_min_diam / self.img_scale[1]
        keep = []
        for idx in range(len(pred_bboxes)):
            dx, dy = get_diams(pred_bboxes[idx])
            if (dx >= min_diameter_x and dy >= min_diameter_y) or pred_scores[idx] == 1: keep.append(idx) 
        pred_bboxes = pred_bboxes[keep]
        pred_scores = pred_scores[keep]
        pred_ids = [pred_ids[i] for i in keep]
        return pred_bboxes, pred_scores, pred_ids

    def update_bboxes_scores(self, shapes_name, new_bboxes, new_scores, new_ids):
        new_bboxes = convert_boxes_from_napari_view(new_bboxes)
        new_scores =  torch.Tensor(list(new_scores))
        new_ids = list(new_ids)
        # if run hasn't been run
        if shapes_name not in self.pred_bboxes.keys():
            self.pred_bboxes[shapes_name] = new_bboxes
            self.pred_scores[shapes_name] = new_scores
            self.pred_ids[shapes_name] = new_ids
        elif len(new_ids)==0: return
        else:
            min_diameter_x = self.cur_min_diam / self.img_scale[0]
            min_diameter_y = self.cur_min_diam / self.img_scale[1]
            # find ids that do are not in self.pred_ids but are in new_ids
            added_box_ids = list(set(new_ids).difference(self.pred_ids[shapes_name]))
            if len(added_box_ids) > 0:
                added_ids = [new_ids.index(box_id) for box_id in added_box_ids]
                #  and add them
                self.pred_bboxes[shapes_name] = torch.cat((self.pred_bboxes[shapes_name], new_bboxes[added_ids]))
                self.pred_scores[shapes_name] = torch.cat((self.pred_scores[shapes_name], new_scores[added_ids]))
                new_ids_to_add = [new_ids[i] for i in added_ids]
                self.pred_ids[shapes_name].extend(new_ids_to_add)
            
            # and find ids that are in self.pred_ids and not in new_ids
            potential_removed_box_ids = list(set(self.pred_ids[shapes_name]).difference(new_ids))
            if len(potential_removed_box_ids) > 0:
                potential_removed_ids = [self.pred_ids[shapes_name].index(box_id) for box_id in potential_removed_box_ids]
                remove_ids = []
                for idx in potential_removed_ids:
                    dx, dy  = get_diams(self.pred_bboxes[shapes_name][idx])
                    if self.pred_scores[shapes_name][idx] > self.cur_confidence and dx > min_diameter_x and dy > min_diameter_y:
                        remove_ids.append(idx)
                # and remove them
                for idx in reversed(remove_ids):
                    self.pred_bboxes[shapes_name] = torch.cat((self.pred_bboxes[shapes_name][:idx, :], self.pred_bboxes[shapes_name][idx+1:, :]))
                    self.pred_scores[shapes_name] = torch.cat((self.pred_scores[shapes_name][:idx], self.pred_scores[shapes_name][idx+1:]))
                    new_pred_ids = self.pred_ids[shapes_name][:idx]
                    new_pred_ids.extend(self.pred_ids[shapes_name][idx+1:])
                    self.pred_ids[shapes_name] = new_pred_ids

    def update_next_id(self, shapes_name, c=0):
        if c!=0:
            self.next_id[shapes_name] = c
        else: self.next_id[shapes_name] += 1

