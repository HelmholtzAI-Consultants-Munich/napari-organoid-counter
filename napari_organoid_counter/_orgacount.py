import torch
from torchvision.transforms import ToTensor

from urllib.request import urlretrieve
from napari.utils import progress

from napari_organoid_counter._utils import *
from napari_organoid_counter import settings


class OrganoiDL():
    '''
    The back-end of the organoid counter widget
    Attributes
    ----------
        device: torch.device
            The current device, either 'cpu' or 'gpu:0'
        cur_confidence: float
            The confidence threshold of the model
        cur_min_diam: float
            The minimum diameter of the organoids
        transfroms: torchvision.transforms.ToTensor
            The transformation for converting numpy image to tensor so it can be given as an input to the model
        model: frcnn
            The Faster R-CNN model
        img_scale: list of floats
            A list holding the image resolution in x and y
        pred_bboxes: dict
            Each key will be a set of predictions of the model, either past or current, and values will be the numpy arrays 
            holding the predicted bounding boxes
        pred_scores: dict
            Each key will be a set of predictions of the model and the values will hold the confidence of the model for each
            predicted bounding box
        pred_ids: dict
            Each key will be a set of predictions of the model and the values will hold the box id for each
            predicted bounding box
        next_id: dict
            Each key will be a set of predictions of the model and the values will hold the next id to be attributed to a 
            newly added box
    '''
    def __init__(self, handle_progress):
        super().__init__()
        
        self.handle_progress = handle_progress
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cur_confidence = 0.05
        self.cur_min_diam = 30
        self.transfroms = ToTensor()

        self.model = None
        self.img_scale = [0., 0.]
        self.pred_bboxes = {}
        self.pred_scores = {}
        self.pred_ids = {}
        self.next_id = {}

    def set_scale(self, img_scale):
        ''' Set the image scale: used to calculate real box sizes. '''
        self.img_scale = img_scale

    def set_model(self, model_name):
        ''' Initialise  model instance and load model checkpoint and send to device. '''
        self.model = frcnn(num_classes=2, rpn_score_thresh=0, box_score_thresh = self.cur_confidence)
        self.load_model_checkpoint(model_name)
        self.model = self.model.to(self.device)

    def download_model(self, model='default'):
        ''' Downloads the model from zenodo and stores it in settings.MODELS_DIR '''
        # specify the url of the file which is to be downloaded
        down_url = settings.MODELS[model]["source"]
        # specify save location where the file is to be saved
        save_loc = join_paths(str(settings.MODELS_DIR), settings.MODELS[model]["filename"])
        # Downloading using urllib
        urlretrieve(down_url,save_loc, self.handle_progress)

    def load_model_checkpoint(self, model_name):
        ''' Loads the model checkpoint for the model specified in model_name '''
        model_checkpoint = join_paths(settings.MODELS_DIR, settings.MODELS[model_name]["filename"])
        ckpt = torch.load(model_checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt) #.state_dict())

    def sliding_window(self,
                       test_img,
                       step,
                       window_size,
                       rescale_factor,
                       prepadded_height,
                       prepadded_width,
                       pred_bboxes=[],
                       scores_list=[]):
        ''' Runs sliding window inference and returns predicting bounding boxes and confidence scores for each box.
        Inputs
        ----------
        test_img: Tensor of size [B, C, H, W]
            The image ready to be given to model as input
        step: int
            The step of the sliding window, same in x and y
        window_size: int
            The sliding window size, same in x and y
        rescale_factor: float
            The rescaling factor by which the image has already been resized. Is 1/downsampling
        prepadded_height: int
            The image height before padding was applied
        prepadded_width: int
            The image width before padding was applied
        pred_bboxes: list of
            The
        scores_list: list of
            The
        Outputs
        ----------
        pred_bboxes: list of Tensors, default is an empty list
            The  resulting predicted boxes are appended here - if model is run at different window
            sizes and downsampling this list will store results of all runs of the sliding window
            so will not be empty the second, third etc. time.
        scores_list: list of Tensor, default is an empty list
            The  resulting confidence scores of the model for the predicted boxes are appended here 
            Same as pred_bboxes, can be empty on first run but stores results of all runs.
        '''
        for i in progress(range(0, prepadded_height, step)):
            for j in progress(range(0, prepadded_width, step)):
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
        ''' Runs inference for an image at multiple window sizes and downsampling rates using sliding window ineference.
        The results are filtered using the NMS algorithm and are then stored to dicts.
        Inputs
        ----------
        img: Numpy array of size [H, W]
            The image ready to be given to model as input
        shapes_name: str
            The name of the new predictions
        window_size: list of ints
            The sliding window size, same in x and y, if multiple sliding window will run mulitple times
        downsampling_sizes: list of ints
            The downsampling factor of the image, list size must match window_size
        window_overlap: float
            The window overlap for the sliding window inference.
        ''' 
        bboxes = []
        scores = []
        # run for all window sizes
        for window_size, downsampling in zip(window_sizes, downsampling_sizes):
            # compute the step for the sliding window, based on window overlap
            rescale_factor = 1 / downsampling
            # window size after rescaling
            window_size = round(window_size * rescale_factor)
            step = round(window_size * window_overlap)
            # prepare image for model - norm, tensor, etc.
            ready_img, prepadded_height, prepadded_width  = prepare_img(img,
                                                                        step,
                                                                        window_size,
                                                                        rescale_factor,
                                                                        self.transfroms,
                                                                        self.device)
            # and run sliding window over whole image
            bboxes, scores = self.sliding_window(ready_img,
                                                 step,
                                                 window_size,
                                                 rescale_factor,
                                                 prepadded_height,
                                                 prepadded_width,
                                                 bboxes,
                                                 scores)
        # stack results
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
        """ After results have been stored in dict this function will filter the dicts based on the confidence
        and min_diameter_um thresholds for the given results defined by shape_name and return the filtered dicts. """
        self.cur_confidence = confidence
        self.cur_min_diam = min_diameter_um
        pred_bboxes, pred_scores, pred_ids = self._apply_confidence_thresh(shapes_name)
        if pred_bboxes.size(0)!=0:
            pred_bboxes, pred_scores, pred_ids = self._filter_small_organoids(pred_bboxes, pred_scores, pred_ids)
        pred_bboxes = convert_boxes_to_napari_view(pred_bboxes)
        return pred_bboxes, pred_scores, pred_ids

    def _apply_confidence_thresh(self, shapes_name):
        """ Filters out results of shapes_name based on the current confidence threshold. """
        if shapes_name not in self.pred_bboxes.keys(): return torch.empty((0))
        keep = (self.pred_scores[shapes_name]>self.cur_confidence).nonzero(as_tuple=True)[0]
        result_bboxes = self.pred_bboxes[shapes_name][keep]
        result_scores = self.pred_scores[shapes_name][keep]
        result_ids = [self.pred_ids[shapes_name][int(i)] for i in keep.tolist()]
        return result_bboxes, result_scores, result_ids
    
    def _filter_small_organoids(self, pred_bboxes, pred_scores, pred_ids):
        """ Filters out small result boxes of shapes_name based on the current min diameter size. """
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
        ''' Updated the results dicts, self.pred_bboxes, self.pred_scores and self.pred_ids with new results.
        If the shapes name doesn't exist as a key in the dicts the results are added with the new key. If the
        key exists then new_bboxes, new_scores and new_ids are compared to the class result dicts and the dicts 
        are updated, either by adding some box (user added box) or removing some box (user deleted a prediction).'''
        
        new_bboxes = convert_boxes_from_napari_view(new_bboxes)
        new_scores =  torch.Tensor(list(new_scores))
        new_ids = list(new_ids)
        # if run hasn't been run
        if shapes_name not in self.pred_bboxes.keys():
            self.pred_bboxes[shapes_name] = new_bboxes
            self.pred_scores[shapes_name] = new_scores
            self.pred_ids[shapes_name] = new_ids
            self.next_id[shapes_name] = len(new_ids)+1

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
        """ Updates the next id to append to result dicts. If input c is given then that will be the next id. """
        if c!=0:
            self.next_id[shapes_name] = c
        else: self.next_id[shapes_name] += 1

    def remove_shape_from_dict(self, shapes_name):
        """ Removes results of shapes_name from all result dicts. """
        del self.pred_bboxes[shapes_name]
        del self.pred_scores[shapes_name]
        del self.pred_ids[shapes_name]
        del self.next_id[shapes_name]
