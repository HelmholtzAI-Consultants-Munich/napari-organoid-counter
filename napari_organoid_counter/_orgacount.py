import os
from urllib.request import urlretrieve
from napari.utils import progress

from napari_organoid_counter._utils import *
from napari_organoid_counter import settings



import torch
import onnxruntime as ort

import time
from contextlib import contextmanager
import cv2

 # to remove
 #update_version_in_mmdet_init_file('mmdet', '2.2.0', '2.3.0')
import mmdet
from mmdet.apis import DetInferencer

def get_best_provider():
    # List all available providers on this machine
    available_providers = ort.get_available_providers()
    # print("Available ONNX Runtime providers:", available_providers)

    # Priority order: CUDA > CoreML > CPU
    if "CUDAExecutionProvider" in available_providers:
        return "CUDAExecutionProvider"
    elif "CoreMLExecutionProvider" in available_providers:
        return "CoreMLExecutionProvider"
    else:
        return "CPUExecutionProvider"

@contextmanager
def profile_section(name, stats_dict):
    """Context manager for timing code sections"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    stats_dict[name] = stats_dict.get(name, 0) + elapsed
    stats_dict[f"{name}_count"] = stats_dict.get(f"{name}_count", 0) + 1

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
        model: ort.InferenceSession
            The detection model loaded from an onnx checkpoint
        model_expect_size: tuple
            The size of the input image the model expects
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

        self.cancel_requested = False

        self.model = None
        self.model_name = None
        self.model_type = 'mmdet' # to remove
        self.model_expect_size = (416, 416)
        self.input_name = None
        self.output_names = None
        self.img_scale = [0., 0.]
        self.pred_bboxes = {}
        self.pred_scores = {}
        self.pred_labels = {}
        self.pred_ids = {}
        self.next_id = {}

    def set_scale(self, img_scale):
        ''' Set the image scale: used to calculate real box sizes. '''
        self.img_scale = img_scale

    def set_model(self, model_name):
        ''' Initialise  model instance and load model checkpoint and send to device. '''

        self.model_name = model_name  # Store the model name
        model_checkpoint = join_paths(str(settings.MODELS_DIR), settings.MODELS[model_name]["filename"])
        if model_checkpoint.endswith('.onnx'):
            provider = get_best_provider()
            self.model = ort.InferenceSession(model_checkpoint, providers=[provider])
            # Get input/output names
            self.model_type = 'onnx'
            self.input_name = self.model.get_inputs()[0].name
            self.output_names = [o.name for o in self.model.get_outputs()]
            
            # Get model-specific preprocessing parameters
            mmdet_path = os.path.dirname(mmdet.__file__)
            base_model_name = settings.CONFIG_MAP.get(model_name, model_name)  # Use mapping
            config_dst = join_paths(mmdet_path, str(settings.CONFIGS[base_model_name]["destination"]))
            pth_checkpoint = model_checkpoint.replace('.onnx', '.pth')
            
            if not os.path.exists(config_dst):
                urlretrieve(settings.CONFIGS[model_name]["source"], config_dst, self.handle_progress)
            
            self.input_size, self.mean, self.std = get_model_preprocessing_params(config_dst, pth_checkpoint)
            self.model_expect_size = self.input_size
            print(f"ONNX Model loaded: {model_name}")
            print(f"  Input size: {self.input_size}")
            print(f"  Mean: {self.mean}")
            print(f"  Std: {self.std}")
        else:
            mmdet_path = os.path.dirname(mmdet.__file__)
            config_dst = join_paths(mmdet_path, str(settings.CONFIGS[model_name]["destination"]))
            # download the corresponding config if it doesn't exist already
            if not os.path.exists(config_dst):
                urlretrieve(settings.CONFIGS[model_name]["source"], config_dst, self.handle_progress)
            self.model = DetInferencer(config_dst, model_checkpoint, self.device, show_progress=False)
            self.model_type = 'mmdet'

    def download_model(self, model_name='yolov3'):
        ''' Downloads the model from zenodo and stores it in settings.MODELS_DIR '''
        # specify the url of the model which is to be downloaded
        down_url = settings.MODELS[model_name]["source"]
        # specify save location where the file is to be saved
        save_loc = join_paths(str(settings.MODELS_DIR), settings.MODELS[model_name]["filename"])
        # downloading using urllib
        urlretrieve(down_url, save_loc, self.handle_progress)

    def sliding_window(self,
                    test_img,
                    step,
                    window_size,
                    rescale_factor,
                    prepadded_height,
                    prepadded_width,
                    pred_bboxes=[],
                    scores_list=[], 
                    labels_list=[]):
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
        pred_bboxes: List
            The list of existing predictions
        scores_list: List
            The list of existing confidence scores for corresponding boxes
        labels_list: List
            The list of existing labels for corresponding boxes
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
        # Profiling variables
        start_time = time.perf_counter()
        profiling_stats = {}

        target_h, target_w = self.model_expect_size

        # go across entire image using sliding window approach with a given window size and step
        for i in progress(range(0, prepadded_height, step)):            
            for j in progress(range(0, prepadded_width, step)):
                if self.cancel_requested:
                    print("Cancellation requested, stopping inference...")
                    return pred_bboxes, scores_list, labels_list  # Return what we have so far   
                if self.model_type == 'onnx':
                    img_crop = test_img[:, :, i:(i+window_size), j:(j+window_size)]

                    # Match MMDet's keep_ratio=True: fit crop within model_expect_size preserving aspect ratio
                    _, _, crop_h, crop_w = img_crop.shape
                    scale_factor = min(target_w / crop_w, target_h / crop_h)
                    new_h, new_w = int(crop_h * scale_factor), int(crop_w * scale_factor)
                    resize_factor_x = window_size / new_h
                    resize_factor_y = window_size / new_w

                    with profile_section('resize', profiling_stats):
                        # Convert float [0,1] BCHW → uint8 HWC to match MMDet's cv2.INTER_LINEAR on uint8
                        img_hwc_uint8 = (img_crop[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                        img_resized = cv2.resize(img_hwc_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        img_crop = img_resized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)  # HWC → BCHW [0,255]

                    # Apply model-specific normalization (img_crop already in [0,255])
                    mean_bchw = self.mean.reshape(1, 3, 1, 1).astype(np.float32)
                    std_bchw = self.std.reshape(1, 3, 1, 1).astype(np.float32)
                    img_crop = (img_crop - mean_bchw) / std_bchw
                    img_crop = img_crop.astype(np.float32)

                    # Run inference
                    outputs = self.model.run(self.output_names, {self.input_name: img_crop})
                    dets, labels = outputs  # dets, labels
                    dets = dets[0]
                    labels = labels[0]
                    if dets.shape[0] == 0:
                        continue
                    else:
                        for bbox_id in range(dets.shape[0]):
                            y1, x1, y2, x2, score = dets[bbox_id]
                            x1 *= resize_factor_x
                            x2 *= resize_factor_x
                            y1 *= resize_factor_y
                            y2 *= resize_factor_y
                            x1_real = torch.div(x1 + i, rescale_factor, rounding_mode='floor')
                            x2_real = torch.div(x2 + i, rescale_factor, rounding_mode='floor')
                            y1_real = torch.div(y1 + j, rescale_factor, rounding_mode='floor')
                            y2_real = torch.div(y2 + j, rescale_factor, rounding_mode='floor')
                            pred_bboxes.append(torch.Tensor([x1_real, y1_real, x2_real, y2_real]))
                            scores_list.append(score)
                            labels_list.append(labels[bbox_id])
                else:
                    # crop
                    img_crop = test_img[i:(i+window_size), j:(j+window_size)]
                    # get predictions
                    output = self.model(img_crop, pred_score_thr=0.05)
                    preds = output['predictions'][0]['bboxes']
                    if len(preds) != 0:
                        for bbox_id in range(len(preds)):
                            y1, x1, y2, x2 = preds[bbox_id]
                            x1_real = torch.div(x1 + i, rescale_factor, rounding_mode='floor')
                            x2_real = torch.div(x2 + i, rescale_factor, rounding_mode='floor')
                            y1_real = torch.div(y1 + j, rescale_factor, rounding_mode='floor')
                            y2_real = torch.div(y2 + j, rescale_factor, rounding_mode='floor')
                            pred_bboxes.append(torch.Tensor([x1_real, y1_real, x2_real, y2_real]))
                            scores_list.append(output['predictions'][0]['scores'][bbox_id])
                            labels_list.append(output['predictions'][0]['labels'][bbox_id])

        # Calculate and print profiling results
        total_time = time.perf_counter() - start_time
        resize_time = profiling_stats.get('resize', 0)
        resize_count = profiling_stats.get('resize_count', 0)
        resize_percentage = (resize_time / total_time * 100) if total_time > 0 else 0

        print(f"\n{'='*60}")
        print(f"Profiling Results for sliding_window()")
        print(f"{'='*60}")
        print(f"Total function time:        {total_time:.4f} seconds")
        print(f"resize_keep_ratio_numpy:")
        print(f"  - Total time:             {resize_time:.4f} seconds")
        print(f"  - Percentage of total:    {resize_percentage:.2f}%")
        print(f"  - Number of calls:        {resize_count}")
        print(f"  - Average time per call:  {resize_time/resize_count:.6f} seconds" if resize_count > 0 else "  - Average time per call:  N/A")
        print(f"{'='*60}\n")

        return pred_bboxes, scores_list, labels_list

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
        labels = []
        # run for all window sizes
        for window_size, downsampling in zip(window_sizes, downsampling_sizes):
            # compute the step for the sliding window, based on window overlap
            rescale_factor = 1 / downsampling
            # window size after rescaling
            window_size = round(window_size * rescale_factor)
            step = round(window_size * window_overlap)
            # prepare image for model - norm, tensor, etc.
            if self.model_type=='onnx':
                ready_img, prepadded_height, prepadded_width  = prepare_img_onnx(img,
                                                                            step,
                                                                            window_size,
                                                                            rescale_factor)
            else:
                ready_img, prepadded_height, prepadded_width  = prepare_img(img,
                                                                            step,
                                                                            window_size,
                                                                            rescale_factor)          
            # and run sliding window over whole image
            bboxes, scores, labels = self.sliding_window(ready_img,
                                                 step,
                                                 window_size,
                                                 rescale_factor,
                                                 prepadded_height,
                                                 prepadded_width,
                                                 bboxes,
                                                 scores, 
                                                 labels)
        
        # if no predictions, store empty tensors so that downstream code still works
        if len(bboxes) == 0:
            self.pred_bboxes[shapes_name] = torch.empty((0, 4))
            self.pred_scores[shapes_name] = torch.empty((0,))
            self.pred_labels[shapes_name] = torch.empty((0,), dtype=torch.long)
            self.pred_ids[shapes_name] = []
            self.next_id[shapes_name] = 1
            return

        # stack results
        bboxes = torch.stack(bboxes)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        # apply NMS to remove overlaping boxes
        bboxes, pred_scores, pred_labels = apply_nms(bboxes, scores, labels)
        # For Detection Only models, set all boxes to class -1 (uncertain/unassigned)
        # For classification models (BC/multiclass), keep the predicted classes
        if self.model_name and "(DO)" in self.model_name:  # TODO: what about the annotation mode?
            pred_labels = torch.full_like(pred_labels, -1)

        self.pred_bboxes[shapes_name] = bboxes
        self.pred_scores[shapes_name] = pred_scores
        self.pred_labels[shapes_name] = pred_labels
        num_predictions = bboxes.size(0)
        self.pred_ids[shapes_name] = [(i+1) for i in range(num_predictions)]
        self.next_id[shapes_name] = num_predictions+1

    def apply_params(self, shapes_name, confidence, min_diameter_um, model_name):
        """ After results have been stored in dict this function will filter the dicts based on the confidence
        and min_diameter_um thresholds for the given results defined by shape_name and return the filtered dicts. """
        self.cur_confidence = confidence
        self.cur_min_diam = min_diameter_um
        pred_bboxes, pred_scores, pred_labels, pred_ids = self._apply_confidence_thresh(shapes_name, model_name)

        # If we are using binary classification (yolov3 (BC)), mark low-confidence boxes as uncertain
        if model_name== 'yolov3 (BC)':
           for idx, score in enumerate(pred_scores):
                if score <= settings.CONFIDENCE_THRESHOLD_CLASS: # TODO: check what the score refers to: objectiness or class confidence?
                    pred_labels[idx] = -1

        # Filter small organoids based on diameter after labeling uncertain predictions
        if pred_bboxes.size(0)!=0:
            pred_bboxes, pred_scores, pred_labels, pred_ids = self._filter_small_organoids(pred_bboxes, pred_scores, pred_labels, pred_ids)
        pred_bboxes = convert_boxes_to_napari_view(pred_bboxes)
        return pred_bboxes, pred_scores, pred_labels, pred_ids

    def _apply_confidence_thresh(self, shapes_name, model_name):
        """ Filters out results of shapes_name based on the current confidence threshold. """
        if shapes_name not in self.pred_bboxes.keys(): return torch.empty((0))

        # Apply confidence threshold
        keep = (self.pred_scores[shapes_name] > self.cur_confidence).nonzero(as_tuple=True)[0]
        result_bboxes = self.pred_bboxes[shapes_name][keep]
        result_scores = self.pred_scores[shapes_name][keep]
        result_labels = self.pred_labels[shapes_name][keep]
        result_ids = [self.pred_ids[shapes_name][int(i)] for i in keep.tolist()]

        # Ensure next_id remains monotonic by setting it to one higher than the max kept ID
        self.next_id[shapes_name] = max(self.pred_ids[shapes_name], default=0) + 1

        return result_bboxes, result_scores, result_labels, result_ids

    def _filter_small_organoids(self, pred_bboxes, pred_scores, pred_labels, pred_ids):
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
        pred_labels = pred_labels[keep]
        pred_ids = [pred_ids[i] for i in keep]
        return pred_bboxes, pred_scores, pred_labels, pred_ids

    def update_bboxes_scores(self, shapes_name, new_bboxes, new_scores, new_labels, new_ids):
        ''' Updated the results dicts, self.pred_bboxes, self.pred_scores and self.pred_ids with new results.
        If the shapes name doesn't exist as a key in the dicts the results are added with the new key. If the
        key exists then new_bboxes, new_scores and new_ids are compared to the class result dicts and the dicts 
        are updated, either by adding some box (user added box) or removing some box (user deleted a prediction).'''
        
        new_bboxes = convert_boxes_from_napari_view(new_bboxes)
        new_scores =  torch.Tensor(list(new_scores))
        new_labels = torch.Tensor(list(new_labels))
        new_ids = list(new_ids)
        # if run hasn't been run
        if shapes_name not in self.pred_bboxes.keys():
            self.pred_bboxes[shapes_name] = new_bboxes
            self.pred_scores[shapes_name] = new_scores
            self.pred_labels[shapes_name] = new_labels
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
                self.pred_labels[shapes_name] = torch.cat((self.pred_labels[shapes_name], new_labels[added_ids]))
                new_ids_to_add = [new_ids[i] for i in added_ids]
                self.pred_ids[shapes_name].extend(new_ids_to_add)
            
            # Update existing boxes that have been modified (resized or class changed)
            # For each box_id that exists in both old and new, update its geometry and label
            common_box_ids = list(set(new_ids).intersection(self.pred_ids[shapes_name]))
            for box_id in common_box_ids:
                new_idx = new_ids.index(box_id)
                old_idx = self.pred_ids[shapes_name].index(box_id)
                # Update bbox coordinates (handles resizing)
                self.pred_bboxes[shapes_name][old_idx] = new_bboxes[new_idx]
                # Update scores (in case user modified manually added boxes)
                self.pred_scores[shapes_name][old_idx] = new_scores[new_idx]
                # Update labels (handles class assignment changes)
                self.pred_labels[shapes_name][old_idx] = new_labels[new_idx]
            
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
                    self.pred_labels[shapes_name] = torch.cat((self.pred_labels[shapes_name][:idx], self.pred_labels[shapes_name][idx+1:]))
                    new_pred_ids = self.pred_ids[shapes_name][:idx]
                    new_pred_ids.extend(self.pred_ids[shapes_name][idx+1:])
                    self.pred_ids[shapes_name] = new_pred_ids

    def update_next_id(self, shapes_name, c=0):
        """ Updates the next id to append to result dicts. If input c is given then that will be the next id. """
        if c!=0:
            self.next_id[shapes_name] = c
        else: 
            # Reset next_id to one higher than the current max ID (or 1 if no boxes remain)
            self.next_id[shapes_name] = max(self.pred_ids[shapes_name], default=0) + 1

    def remove_shape_from_dict(self, shapes_name):
        """ Removes results of shapes_name from all result dicts. """
        del self.pred_bboxes[shapes_name]
        del self.pred_scores[shapes_name]
        del self.pred_labels[shapes_name]
        del self.pred_ids[shapes_name]
        del self.next_id[shapes_name]

    def rename_shape_key(self, old_name: str, new_name: str):
        """Rename a prediction set across all internal dicts."""
        for d in (self.pred_bboxes, self.pred_scores, self.pred_labels, self.pred_ids, self.next_id):
            if old_name in d and new_name not in d:
                d[new_name] = d.pop(old_name)
            elif old_name in d and new_name in d:
                # merge conservatively: prefer existing 'new_name' and drop 'old_name'
                d.pop(old_name)
