import torch
from typing import List
import math

from skimage.io import imsave
from datetime import datetime

import napari

from napari import layers
from napari.utils.notifications import show_info, show_error, show_warning

import numpy as np

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QApplication, QDialog, QFileDialog, QGroupBox, 
    QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QProgressBar, 
    QSlider, QCheckBox, QScrollArea, QTreeWidget, QTreeWidgetItem
)

from napari_organoid_counter._orgacount import OrganoiDL
from napari_organoid_counter import _utils as utils
from napari_organoid_counter import settings

import warnings
warnings.filterwarnings("ignore")


class OrganoidCounterWidget(QWidget):
    '''
    The main widget of the organoid counter
    Parameters
    ----------
        napari_viewer: string
            The current napari viewer
        window_sizes: list of ints, default [1024]
            A list with the sizes of the windows on which the model will be run. If more than one window_size is given then the model will run on several window sizes and then 
            combine the results
        downsampling:list of ints, default [2]
            A list with the sizes of the downsampling ratios for each window size. List size must be the same as the window_sizes list
        min_diameter: int, default 30
            The minimum organoid diameter given in um
        confidence: float, default 0.8
            The model confidence threhsold - equivalent to box_score_thresh of faster_rcnn
    Attributes
    ----------
        model_name: str
            The name of the model user has selected
        image_layer_names: list of strings
            Will hold the names of all the currently open images in the viewer
        image_layer_name: string
            The image we are currently working on
        shape_layer_names: list of strings
            Will hold the names of all the currently open images in the viewer
        save_layer_name: string
            The name of the shapes layer that has been selected for saving
        cur_shapes_name: string
            The name of the shapes layer that has been selected for visualisation
        cur_shapes_layer: napari.layers.Shapes
            The current shapes layer we are working on - it's name should correspond to cur_shapes_name
        organoiDL: OrganoiDL
            The class in which all the computations are performed for computing and storing the organoids bounding boxes and confidence scores
        num_organoids: int
            The current number of organoids
        original_images: dict
        original_contrast: dict
    '''
    def __init__(self, 
                napari_viewer,
                window_sizes: List = [1024],
                downsampling: List = [2],
                window_overlap: float = 0.5,
                min_diameter: int = 30,
                confidence: float = 0.8):
        super().__init__()

        # assign class variables
        self.viewer = napari_viewer 

        # create cache dir for models if it doesn't exist and add any previously added local
        # models to the model dict
        settings.init()
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        utils.add_local_models()
        self.model_id = 2 # yolov3
        self.model_name = list(settings.MODELS.keys())[self.model_id]
        
        # init params 
        self.window_sizes = window_sizes
        self.downsampling = downsampling
        self.window_overlap = window_overlap
        self.min_diameter = min_diameter
        self.confidence = confidence

        self.image_layer_names = []
        self.image_layer_name = None 
        self.shape_layer_names = []
        self.save_layer_name = ''
        self.cur_shapes_name = ''
        self.cur_shapes_layer = None
        self.num_organoids = 0
        self.original_images = {}
        self.original_contrast = {}
        self.stored_confidences = {}
        self.stored_diameters = {}

        # Annotation Mode 
        self.annotation_mode = 0 # Default to Detection Only mode (0)

        # Mapping annotation modes to names and valid class sets
        self.annotation_mode_mapping = {
            0: {"name": "Detection Only (DO)", "classes": {0}},
            1: {"name": "Binary Classification (BC)", "classes": {0, 1}},
        }

        # Auto-generate mappings for 3 to 10 classes
        for n in range(2, 10):
            self.annotation_mode_mapping[n] = {
                "name": f"{n+1} Classes",
                "classes": set(range(n+1)),
            }

        # Mapping class numbers to colors
        self.color_mapping = {
            0: (settings.COLOR_CLASS_0, "Green"),
            1: (settings.COLOR_CLASS_1, "Blue"),
            2: (settings.COLOR_CLASS_2, "Orange"),
            3: (settings.COLOR_CLASS_3, "Purple"),
            4: (settings.COLOR_CLASS_4, "Cyan"),
            5: (settings.COLOR_CLASS_5, "Red"),
            6: (settings.COLOR_CLASS_6, "Brown"),
            7: (settings.COLOR_CLASS_7, "Pink"),
            8: (settings.COLOR_CLASS_8, "Yellow"),
            9: (settings.COLOR_CLASS_9, "Light Blue")
        }

        self.selected_classes = self.annotation_mode_mapping[self.annotation_mode]["classes"]  # active class set for current mode
        self.class_count_labels: dict[int, QLabel] = {}  # per-class count labels in legend
        self.class_checkboxes: dict[int, QCheckBox] = {}   # per-class filter checkboxes
        self.master_class_checkbox: QCheckBox | None = None  # "All classes" toggle checkbox
        self.visible_classes_filter: set[int] = set(self.selected_classes)  # start with all visible
        self._shape_name_by_id: dict[int, str] = {}
        self._hover_idx: int | None = None  # current hovered box index
        self._hover_base: str = ""  # cached static hover message (ID, diameters, etc.)

        # Data Browser state
        self.data_folder: str = ""
        self.image_files: list = []  # All discovered image paths
        self.current_image_path: Path | None = None
        self.file_tree: QTreeWidget | None = None

        # Supported image extensions
        self.supported_image_extensions = {'.tif', '.TIF', '.tiff', '.TIFF', '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG'}

        # Setup GUI        
        # Create a container widget for all content
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(self._setup_input_widget())
        container_layout.addWidget(self._setup_output_widget())
        container_layout.addWidget(self._setup_data_browser_widget())
        container_layout.addStretch(1)  # Push everything to top

        # Wrap in scroll area
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        # initialise organoidl instance
        self.organoiDL = OrganoiDL(self.handle_progress)

        # get already opened layers
        self.image_layer_names = self._get_layer_names()
        if len(self.image_layer_names)>0: self._update_added_image(self.image_layer_names)
        self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        if len(self.shape_layer_names)>0: self._update_added_shapes(self.shape_layer_names)

        # for name in self.shape_layer_names:
        #     layer = self.viewer.layers[name]
        #     self._shape_name_by_id[id(layer)] = layer.name
        #     layer.events.name.connect(lambda e, layer=layer: self._on_shapes_layer_renamed(layer))

        # and watch for newly added images or shapes
        self.viewer.layers.events.inserted.connect(self._added_layer)
        self.viewer.layers.events.removed.connect(self._removed_layer)
        self.viewer.layers.selection.events.changed.connect(self._sel_layer_changed)
    
        # setup flags used for changing slider and text of min diameter and confidence threshold
        self.diameter_slider_changed = False 
        self.confidence_slider_changed = False

        # Key binding to reset the edge_color of selected bounding boxes to the original magenta color
        @self.viewer.bind_key('m', overwrite=True)
        def change_to_original_color(viewer: napari.Viewer):
            if self.cur_shapes_layer is not None:  # Ensure shapes layer exists
                selected_shapes = self.cur_shapes_layer.selected_data
                if len(selected_shapes) > 0:
                    # Modify the edge color only for the selected shapes
                    current_edge_colors = self.cur_shapes_layer.edge_color
                    for idx in selected_shapes:
                        # Revert to the original color
                        current_edge_colors[idx] = settings.COLOR_DEFAULT
                    self.cur_shapes_layer.edge_color = current_edge_colors  # Apply the changes
                    show_info(f"Reset edge color of {list(selected_shapes)} to magenta.")
                    self._refresh_class_counts()
                else:
                    show_warning("No shapes selected to reset edge color.")

        # Activate key bindings for class-color shortcuts
        self.update_key_bindings()
        self._setup_mouse_callback()

    def update_key_bindings(self):
        """ Update key bindings based on selected classes """            

        # Unbind all potential class keys (CTRL+0 to CTRL+9)
        for class_num in range(10):
            key = f'Control-{class_num}'
            if key in self.viewer.keymap:
                self.viewer.keymap.pop(key) # Remove the key binding if it already exists

        # Bind all keys and validate them on press
        bound_keys = []
        for class_num in range(10):
            key = f'Control-{class_num}'
            
            # Capture 'class_num' using a lambda default argument
            @self.viewer.bind_key(key, overwrite=True)
            def change_color_for_class(viewer: napari.Viewer, class_num=class_num):
                # Check if the class is valid for the current annotation mode
                if class_num not in self.selected_classes:
                    show_error(f"Class {class_num} is not available in the current annotation mode.")
                    return
                
                # Ensure we are NOT in detection-only mode
                if self.annotation_mode == 0:
                    show_error(f"Cannot change class in Detection Only annotation mode.")
                    return
                
                # Proceed with the color change if valid
                if self.cur_shapes_layer:
                    selected_shapes = self.cur_shapes_layer.selected_data
                    if selected_shapes:
                        self.change_edge_color(viewer, selected_shapes, class_num)
                    else:
                        show_warning("No shapes selected to change edge color.")
                else:
                    show_warning("No active shapes layer available.")

            # Confirm valid key bindings
            if class_num in self.selected_classes:
                # Suppress message if in detection-only mode
                if self.annotation_mode == 0:
                    return
                bound_keys.append(f"{key} to change to class {class_num}")
            # Display a dynamic message showing all valid key bindings
        if bound_keys:
            mode_name = self.annotation_mode_mapping[self.annotation_mode]["name"]
            binding_message = ", ".join(bound_keys)
            show_info(f"Switched to {mode_name} annotation mode. Use {binding_message}.")

            #show_info(f"Switched to: {self.annotation_mode_mapping[mode]['name']} mode. Use ")
            #show_info(f"Use {key} to change edge color to class {class_num}.")

    def _on_shapes_layer_renamed(self, layer):
        old = self._shape_name_by_id.get(id(layer))
        new = layer.name
        if not old or old == new:
            return

        # 1) migrate OrganoiDL dict keys
        if hasattr(self.organoiDL, "rename_shape_key"):
            self.organoiDL.rename_shape_key(old, new)

        # 2) update widget state dicts
        if old in self.stored_confidences:
            self.stored_confidences[new] = self.stored_confidences.pop(old)
        if old in self.stored_diameters:
            self.stored_diameters[new] = self.stored_diameters.pop(old)

        # 3) update our layer-name lists
        if old in self.shape_layer_names:
            idx = self.shape_layer_names.index(old)
            self.shape_layer_names[idx] = new


        # 5) update active names
        if self.cur_shapes_name == old:
            self.cur_shapes_name = new
        if self.save_layer_name == old:
            self.save_layer_name = new

        # 6) cache the new name
        self._shape_name_by_id[id(layer)] = new

    def change_edge_color(self, viewer: napari.Viewer, selected_shapes, class_num):
        """Change the edge color of selected shapes based on the class number."""

        # Check if the class_num is valid in the mapping
        if class_num in settings.COLOR_MAPPING:
            current_edge_colors = self.cur_shapes_layer.edge_color

            # Update the edge color for the selected shapes
            for idx in selected_shapes:
                current_edge_colors[idx] = settings.COLOR_MAPPING[class_num][0] # Set RGBA color

            # Apply the updated colors back to the layer
            self.cur_shapes_layer.edge_color = current_edge_colors
            show_info(f"Changed edge color of shapes {list(selected_shapes)} to {settings.COLOR_MAPPING[class_num][1]}.") # Print color name
            self._apply_class_filter()
            self._refresh_class_counts()
        else:
            show_warning(f"Class {class_num} has no associated color.")  

    def handle_progress(self, blocknum, blocksize, totalsize):
        """ When the model is being downloaded, this method is called and th progress of the download
        is calculated and displayed on the progress bar. This function was re-implemented from:
        https://www.geeksforgeeks.org/pyqt5-how-to-automate-progress-bar-while-downloading-using-urllib/ """
        read_data = blocknum * blocksize # calculate the progress
        if totalsize > 0:
            download_percentage = read_data * 100 / totalsize
            self.progress_bar.setValue(int(download_percentage))
            QApplication.processEvents()

    def _sel_layer_changed(self, event):
        """ Is called whenever the user selects a different layer to work on. """
        cur_layer_list = list(self.viewer.layers.selection)
        if len(cur_layer_list)==0: return
        cur_seg_selected = cur_layer_list[-1]
        # switch to values of other shapes layer if clicked
        if type(cur_seg_selected)==layers.Shapes:
            if self.cur_shapes_layer is not None:
                self.stored_confidences[self.cur_shapes_name] = self.confidence_slider.value()/100
                self.stored_diameters[self.cur_shapes_name] = self.min_diameter_slider.value()
            self.cur_shapes_layer = cur_seg_selected
            self.cur_shapes_name = cur_seg_selected.name
            # update min diameter text and slider with previous value of that layer
            self.min_diameter = self.stored_diameters[self.cur_shapes_name]
            self.min_diameter_textbox.setText(str(self.min_diameter))
            # update confidence text and slider with previous value of that layer
            self.confidence = self.stored_confidences[self.cur_shapes_name]
            self.confidence_textbox.setText(str(self.confidence))
            self._hover_idx = None
            self._hover_base = ""

    def _added_layer(self, event):
        # get names of added layers, image and shapes
        new_image_layer_names = self._get_layer_names()
        new_shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        new_image_layer_names = [name for name in new_image_layer_names if name not in self.image_layer_names]
        new_shape_layer_names = [name for name in new_shape_layer_names if name not in self.shape_layer_names]
        if len(new_image_layer_names)>0 : 
            self._update_added_image(new_image_layer_names)
            self.image_layer_names.extend(new_image_layer_names)
        if len(new_shape_layer_names)>0:
            self._update_added_shapes(new_shape_layer_names)
            self.shape_layer_names.extend(new_shape_layer_names)

            # for name in new_shape_layer_names:
            #     layer = self.viewer.layers[name]
            #     self._shape_name_by_id[id(layer)] = layer.name
            #     layer.events.name.connect(lambda e, layer=layer: self._on_shapes_layer_renamed(layer))

            # reset edge color
            for name in new_shape_layer_names:
                self.viewer.layers[name].current_edge_color = settings.COLOR_DEFAULT
            
    def _removed_layer(self, event):
        """ Is called whenever a layer has been deleted (by the user) and removes the layer from GUI and backend. """
        new_image_layer_names = self._get_layer_names()
        new_shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        removed_image_layer_names = [name for name in self.image_layer_names if name not in new_image_layer_names]
        removed_shape_layer_names = [name for name in self.shape_layer_names if name not in new_shape_layer_names]
        if len(removed_image_layer_names)>0:
            self._update_removed_image(removed_image_layer_names)
            self.image_layer_names = new_image_layer_names
        if len(removed_shape_layer_names)>0:
            self._update_remove_shapes(removed_shape_layer_names)
            self.shape_layer_names = new_shape_layer_names

    def _preprocess(self):
        """ Preprocess the current image in the viewer to improve visualisation for the user """
        img = self.original_images[self.image_layer_name]
        img = utils.apply_normalization(img)
        self.viewer.layers[self.image_layer_name].data = img
        self.viewer.layers[self.image_layer_name].contrast_limits = (0,255)

    def _update_num_organoids(self, len_bboxes):
        """ Updates the number of organoids displayed in the viewer """
        self.num_organoids = len_bboxes
        new_text = 'Number of organoids: '+str(self.num_organoids)
        self.organoid_number_label.setText(new_text)

    def _update_vis_bboxes(self, bboxes, scores, labels, box_ids, labels_layer_name):
        """ Adds the shapes layer to the viewer or updates it if already there """
        self._update_num_organoids(len(bboxes))

        # Convert PyTorch tensors to lists (if they are tensors)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        # Text parameters (for all models)
        text_params = {
            'string': 'ID: {box_id}\nConf.: {scores:.2f}',
            'size': 9,
            'anchor': 'upper_left',
        }

        # Edge color for the boxes
        all_labels_unassigned = all(int(label) == -1 for label in labels)
        use_default_color = (self.annotation_mode == 0 or "(DO)" in self.model_name) and all_labels_unassigned
        edge_color = utils.get_edge_color(labels, use_default_color)
                    
        # if layer already exists
        if labels_layer_name in self.shape_layer_names: 
            self.viewer.layers[labels_layer_name].data = bboxes # hack to get edge_width stay the same!
            self.viewer.layers[labels_layer_name].properties = {'box_id': box_ids,'scores': scores, 'labels': labels}
            self.viewer.layers[labels_layer_name].edge_color = edge_color
            self.viewer.layers[labels_layer_name].edge_width = 12
            self.viewer.layers[labels_layer_name].refresh()
            self.viewer.layers[labels_layer_name].refresh_text()
        # or if this is the first run
        else:
            # if no organoids were found just make an empty shapes layer
            if self.num_organoids==0: 
                self.cur_shapes_layer = self.viewer.add_shapes(name=labels_layer_name,
                                                               properties={'box_id': [],'scores': [], 'labels': []})
            # otherwise make the layer and add the boxes
            else:
                properties = {'box_id': box_ids,'scores': scores, 'labels': labels}
                text_params = text_params
                self.cur_shapes_layer = self.viewer.add_shapes(bboxes, 
                                                               name=labels_layer_name,
                                                               scale=self.viewer.layers[self.image_layer_name].scale,
                                                               face_color='transparent',  
                                                               properties = properties,
                                                               text = text_params,
                                                               edge_color=edge_color,
                                                               shape_type='rectangle',
                                                               edge_width=12) # warning generated here
                            
            # set current_edge_width so edge width is the same when users annotate - doesnt' fix new preds being added!
            self.viewer.layers[labels_layer_name].current_edge_width = 12
        
        self._apply_class_filter()
        self._refresh_class_counts()

    def _on_preprocess_click(self):
        """ Is called whenever preprocess button is clicked """
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else: self._preprocess()

    def _on_run_click(self):
        """ Is called whenever Run Organoid Counter button is clicked """
        # Reset cancellation flag
        self.organoiDL.cancel_requested = False
        self.cancel_btn.setEnabled(True)

        # Check if the model is 'Binary Classification' and ensure it's not in 'Detection Only' mode
        current_annotation_mode = self.annotation_mode
        # TODO: make the condition more general (applicable to other models as well)
        if self.model_name == 'yolov3 (BC)' and current_annotation_mode == 0:
            show_error("Please switch to Binary Classification Annotation mode or use more than 2 classes to use the Binary Classification model.")
            return
            
        # check if an image has been loaded
        if not self.image_layer_name: 
            show_info('Please load an image first and try again!')
            return
        # check if model exists locally and if not ask user if it's ok to download
        if not utils.return_is_file(settings.MODELS_DIR, settings.MODELS[self.model_name]["filename"]): 
            confirm_window = ConfirmUpload(self)
            confirm_window.exec_()
            # if user clicks cancel return doing nothing 
            if confirm_window.result() != QDialog.Accepted: return
            # otherwise donwload model and display progress in progress bar
            else: 
                self.progress_box.show()
                self.organoiDL.download_model(self.model_name)
                self.progress_box.hide()
        
        # load model checkpoint
        self.organoiDL.set_model(self.model_name)
        if self.organoiDL.img_scale[0]==0: self.organoiDL.set_scale(self.viewer.layers[self.image_layer_name].scale)
        
        # make sure the number of windows and downsamplings are the same
        if len(self.window_sizes) != len(self.downsampling): 
            show_info('Keep number of window sizes and downsampling the same and try again!')
            return
        
        # get the current image 
        img_data = self.viewer.layers[self.image_layer_name].data
        
        # check that image is grayscale
        if len(utils.squeeze_img(img_data).shape) > 2:
            show_info('Only grayscale images currently supported. Try a different image or process it first and try again!')
            return 
        
        # update the viewer with the new bboxes
        labels_layer_name = 'Labels-'+self.image_layer_name
        if labels_layer_name in self.shape_layer_names:
            show_info('Found existing labels layer. Please remove or rename it and try again!')
            return 
        
        # show activity docker for progrgess bar while running 
        self.viewer.window._status_bar._toggle_activity_dock(True)
       
        # run inference
        self.organoiDL.run(img_data, 
                           labels_layer_name,
                           self.window_sizes,
                           self.downsampling,
                           self.window_overlap)
        
        raw_total = self.organoiDL.pred_bboxes[labels_layer_name].size(0)
        print(f"Backend detected {raw_total} boxes (before any filtering).")

        if self.organoiDL.cancel_requested:
            self.cancel_btn.setEnabled(False)
            self.viewer.window._status_bar._toggle_activity_dock(False)
            show_info("Inference cancelled by user")
            return

        # set the confidence threshold, remove small organoids and get bboxes in format o visualise
        bboxes, scores, labels, box_ids = self.organoiDL.apply_params(labels_layer_name, self.confidence, self.min_diameter, self.model_name)
        # hide activcity dock on completion
        self.viewer.window._status_bar._toggle_activity_dock(False)

        # Disable cancel button when done
        self.cancel_btn.setEnabled(False)

        # update widget with results
        self._update_vis_bboxes(bboxes, scores, labels, box_ids, labels_layer_name)
        # and update cur_shapes_name to newly created shapes layer
        self.cur_shapes_name = labels_layer_name
        # preprocess the image if not done so already to improve visualisation
        self._preprocess() 

    def _on_cancel_click(self):
        """ Is called when Cancel button is clicked """
        self.organoiDL.cancel_requested = True
        self.cancel_btn.setEnabled(False)
        show_info("Cancelling inference...")

    def _on_model_selection_changed(self):
        """ Is called when user selects a new model from the dropdown menu. """
        self.model_name = self.model_selection.currentText()

    def _on_choose_model_clicked(self):
        """ Is called whenever browse button is clicked for model selection """
        # called when the user hits the 'browse' button to select a model
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.AnyFile)
        if fd.exec_():
            model_path = fd.selectedFiles()[0]
        import shutil
        shutil.copy2(model_path, settings.MODELS_DIR)
        model_name = utils.add_to_dict(model_path)
        self.model_selection.addItem(model_name)

    def _on_window_sizes_changed(self):
        """ Is called whenever user changes the window sizes text box """
        new_window_sizes = self.window_sizes_textbox.text()
        new_window_sizes = new_window_sizes.split(',')
        self.window_sizes = [int(win_size) for win_size in new_window_sizes]

    def _on_downsampling_changed(self):
        """ Is called whenever user changes the downsampling text box """
        new_downsampling = self.downsampling_textbox.text()
        new_downsampling = new_downsampling.split(',')
        self.downsampling = [int(ds) for ds in new_downsampling]

    def _rerun(self):
        """ Is called whenever user changes one of the two parameter sliders """
        # check if OrganoiDL instance exists - create it if not and set there current boxes, scores and ids        
        if self.organoiDL.img_scale[0]==0: 
            self.organoiDL.set_scale(self.cur_shapes_layer.scale)
        # self.organoiDL.update_next_id(self.cur_shapes_name, len(self.cur_shapes_layer.scale)+1)
        # box_ids = self.cur_shapes_layer.properties['box_id']
        # next_id_val = (int(box_ids.max()) + 1) if box_ids.size else 1
        # self.organoiDL.update_next_id(self.cur_shapes_name, next_id_val)

        # GUARD: ensure backend dicts exist under the current layer name
        if self.cur_shapes_name not in self.organoiDL.pred_bboxes:
            self.organoiDL.pred_bboxes[self.cur_shapes_name] = torch.empty((0, 4))
            self.organoiDL.pred_scores[self.cur_shapes_name] = torch.empty((0,))
            self.organoiDL.pred_labels[self.cur_shapes_name] = torch.empty((0,), dtype=torch.long)
            self.organoiDL.pred_ids[self.cur_shapes_name] = []
            self.organoiDL.next_id[self.cur_shapes_name] = 1

        # make sure to add info to cur_shapes_layer.metadata to differentiate this action from when user adds/removes boxes
        with utils.set_dict_key( self.cur_shapes_layer.metadata, 'napari-organoid-counter:_rerun', True):
            # Derive current labels from edge colors to capture any class assignments user made
            labels_prop, _ = self._assign_labels(validate=False)

            # first update bboxes in organoiDL in case user has added/removed/modified
            self.organoiDL.update_bboxes_scores(self.cur_shapes_name,
                                                self.cur_shapes_layer.data, 
                                                self.cur_shapes_layer.properties['scores'],
                                                labels_prop,
                                                self.cur_shapes_layer.properties['box_id'])
            # and get new boxes, scores and box ids based on new confidence and min_diameter values 
            bboxes, scores, labels, box_ids = self.organoiDL.apply_params(self.cur_shapes_name, self.confidence, self.min_diameter, self.model_name)
            self._update_vis_bboxes(bboxes, scores, labels, box_ids, self.cur_shapes_name)
            self._apply_class_filter()
            self._refresh_class_counts()
            self.organoiDL.update_next_id(self.cur_shapes_name)   # keep counter monotonic

    def _on_diameter_slider_changed(self):
        """ Is called whenever user changes the Minimum Diameter slider """
        # get current value
        self.min_diameter = self.min_diameter_slider.value()
        self.diameter_slider_changed = True
        if int(self.min_diameter_textbox.text())!= self.min_diameter:
            self.min_diameter_textbox.setText(str(self.min_diameter))
        self.diameter_slider_changed = False
        # check if no labels loaded yet
        if len(self.shape_layer_names)==0: return
        self._rerun() 
    
    def _on_diameter_textbox_changed(self):
        """ Is called whenever user changes the minimum diameter from the textbox """
        # check if no labels loaded yet
        if self.diameter_slider_changed: return
        self.min_diameter = int(self.min_diameter_textbox.text())
        if self.min_diameter_slider.value() != self.min_diameter:
            self.min_diameter_slider.setValue(self.min_diameter)
        if len(self.shape_layer_names)==0: return
        self._rerun()

    def _on_confidence_slider_changed(self):
        """ Is called whenever user changes the confidence slider """
        self.confidence = self.confidence_slider.value()/100
        self.confidence_slider_changed = True
        if float(self.confidence_textbox.text()) != self.confidence:
            self.confidence_textbox.setText(str(self.confidence))
        self.confidence_slider_changed = False
        # check if no labels loaded yet
        if len(self.shape_layer_names)==0: return
        self._rerun()

    def _on_confidence_textbox_changed(self):
        """ Is called whenever user changes the confidence value from the textbox """
        if self.confidence_slider_changed: return
        self.confidence = float(self.confidence_textbox.text())
        slider_conf_value = int(self.confidence*100)
        if self.confidence_slider.value() != slider_conf_value:
            self.confidence_slider.setValue(slider_conf_value)
        if len(self.shape_layer_names)==0: return
        self._rerun()

    def _on_image_selection_changed(self):
        """ Is called whenever a new image has been selected from the drop down box """
        self.image_layer_name = self.image_layer_selection.currentText()
    

    def _on_reset_click(self):
        """ Is called whenever Reset Configs button is clicked """
        # reset params
        self.min_diameter = 30
        self.confidence = 0.8
        vis_confidence = int(self.confidence*100)
        self.min_diameter_slider.setValue(self.min_diameter)
        self.confidence_slider.setValue(vis_confidence)
        if self.image_layer_name:
            # reset to original image
            self.viewer.layers[self.image_layer_name].data = self.original_images[self.image_layer_name]
            self.viewer.layers[self.image_layer_name].contrast_limits = self.original_contrast[self.image_layer_name]

    def _on_screenshot_click(self):
        """ Is called whenever Take Screenshot button is clicked """
        screenshot=self.viewer.screenshot()
        if not self.image_layer_name: potential_name = datetime.now().strftime("%d%m%Y%H%M%S")+'screenshot.png'
        else: potential_name = self.image_layer_name+datetime.now().strftime("%d%m%Y%H%M%S")+'_screenshot.png'
        fd = QFileDialog()
        name,_ = fd.getSaveFileName(self, 'Save File', potential_name, 'Image files (*.png);;(*.tiff)') #, 'CSV Files (*.csv)')
        if name: imsave(name, screenshot)

    def on_annotation_mode_changed(self, mode):
        """Callback for dropdown selection."""
        self.annotation_mode = mode
        self.selected_classes = self.annotation_mode_mapping[mode]["classes"]
        #show_info(f"Switched to: {self.annotation_mode_mapping[mode]['name']} mode.")
        self.update_key_bindings()  # Update key bindings based on the selected annotation mode
        self._refresh_color_mapping_box()
    
    def _assign_labels(self, validate=True):
        """ Assign labels to the bounding boxes based on their edge colors.
        
        Args:
            validate: If True, check that all colors are valid and return all_valid flag.
                     If False, assign -1 to unmatched colors (used during _rerun to be lenient).
        """
        # Get the edge colors for all bounding boxes
        edge_colors = self.cur_shapes_layer.edge_color
        all_valid = True  # Flag to track if all bounding boxes are valid
        # Check for annotation mode
        if self.annotation_mode == 0: # Detection Only Mode
            # Set all labels to None since we don't need them in detection-only mode
            labels = [-1] * len(edge_colors)

        else: # For other annotation modes (Binary Classification, 3 classes, 4 classes, etc.)

            # Assign organoid label based on edge_color by matching against COLOR_MAPPING
            labels = []

            for edge_color in edge_colors:
                matched = False # Flag to check if the current color matches any valid color
                
                # First check if it's the default magenta color (unassigned/uncertain)
                if np.allclose(edge_color[:3], settings.COLOR_DEFAULT[:3], rtol=1e-3, atol=1e-3):
                    labels.append(-1)  # Assign -1 for default/unassigned boxes
                    matched = True
                else:
                    # Check against all color mappings to find the actual class number
                    for class_num, (color_rgba, color_name) in settings.COLOR_MAPPING.items():
                        if np.allclose(edge_color[:3], color_rgba[:3], rtol=1e-3, atol=1e-3):
                            labels.append(class_num)  # Assign actual class number
                            matched = True
                            break
                
                # If no match for this edge_color
                if not matched:
                    if validate:
                        all_valid = False
                        break # Exit the loop early if any color is invalid
                    else:
                        # During _rerun, be lenient and assign -1 for unmatched colors
                        labels.append(-1)
        return labels, all_valid

    def _on_save_csv_click(self): 
        """ Is called whenever Save features button is clicked """
        bboxes = self.viewer.layers[self.save_layer_name].data
        if not bboxes: 
            show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
            return
        
        # Get the labels for the bounding boxes
        labels, all_valid = self._assign_labels()
        # If any bounding box has an invalid color, show a warning and return without saving
        if not all_valid:
            # If no match is found, mark as not all boxes are colored
            show_error("Some organoids have not been assigned a valid class (null class). Please ensure all organoids are properly classified before saving.")
            return
        
        # write diameters and area to csv
        data_csv = utils.get_bbox_diameters(bboxes, 
                                        self.viewer.layers[self.save_layer_name].properties['box_id'],
                                        self.viewer.layers[self.save_layer_name].scale,
                                        labels,)
        fd = QFileDialog()
        name, _ = fd.getSaveFileName(self, 'Save File', self.save_layer_name, 'CSV files (*.csv)')#, 'CSV Files (*.csv)')
        if name: utils.write_to_csv(name, data_csv)

    def _on_save_json_click(self):
        """ Is called whenever Save boxes button is clicked """
        bboxes = self.viewer.layers[self.save_layer_name].data
        #scores = #add
        if not bboxes: 
            show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
            return
        
        # Get the labels for the bounding boxes
        labels, all_valid = self._assign_labels()
            
        # If any bounding box has an invalid color, show a warning and return without saving
        if not all_valid:
            # If no match is found, mark as not all boxes are colored
            show_error("Some organoids have not been assigned a valid class (null class). Please ensure all organoids are properly classified before saving.")
            return

        # Get the bounding boxes as a dictionary
        data_json = utils.get_bboxes_as_dict(bboxes, 
                                    self.viewer.layers[self.save_layer_name].properties['box_id'],
                                    self.viewer.layers[self.save_layer_name].properties['scores'],
                                    self.viewer.layers[self.save_layer_name].scale,
                                    labels,)
            
        
        # write bbox coordinates to json
        fd = QFileDialog()
        name,_ = fd.getSaveFileName(self, 'Save File', self.save_layer_name, 'JSON files (*.json)')#, 'CSV Files (*.csv)')
        if name: utils.write_to_json(name, data_json)

    def _update_added_image(self, added_items):
        """
        Update the selection box with new images if images have been added and update the self.original_images and self.original_contrast dicts.
        Set the latest added image to the current working image (self.image_layer_name)
        """
        for layer_name in added_items:
            self.image_layer_selection.addItem(layer_name)
            self.original_images[layer_name] = self.viewer.layers[layer_name].data
            self.original_contrast[layer_name] = self.viewer.layers[layer_name].contrast_limits
        self.image_layer_name = added_items[0]

    def _update_removed_image(self, removed_layers):
        """
        Update the selection box by removing image names if image has been deleted and remove items from self.original_images and self.original_contrast dicts.
        """
        # update drop-down selection box and remove image from dict
        for removed_layer in removed_layers:
            item_id = self.image_layer_selection.findText(removed_layer)
            self.image_layer_selection.removeItem(item_id)
            del self.original_images[removed_layer]
            del self.original_contrast[removed_layer]

    def _setup_mouse_callback(self):
        """Set up a mouse move callback to display box IDs in the status bar."""
        @self.viewer.mouse_move_callbacks.append
        def mouse_move_callback(viewer, event):
            if self.cur_shapes_layer is None:
                self._hover_idx = None
                self._hover_base = ""
                return

            layer = self.cur_shapes_layer
            pos_data = layer.world_to_data(event.position)
            value = layer.get_value(pos_data)

            # If there's no shape under cursor, clear cached index and return
            if value is None or value[0] is None:
                self._hover_idx = None
                self._hover_base = ""
                return

            # Get shape index and properties
            idx = value[0]

            # Recompute the static part ONLY when entering a new box
            if idx != self._hover_idx:
                props = layer.properties
                n = len(layer.data)
                box_id = props.get('box_id', [None]*n)[idx]
                scores = props.get('scores', [None]*n)
                score = scores[idx] if idx < len(scores) else None
                score_txt = f", Conf.: {score:.2f}" if score is not None else ""

                bbox = layer.data[idx]                     # [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
                sx, sy = layer.scale
                d1 = abs(bbox[2][0] - bbox[0][0]) * sx     # µm
                d2 = abs(bbox[2][1] - bbox[0][1]) * sy     # µm
                area = math.pi * d1 * d2 / 4               # µm²

                self._hover_base = (
                    f"Organoid ID: {box_id}{score_txt}, "
                    f"d1={d1:.1f} µm, d2={d2:.1f} µm, area={area:.1f} µm² | Coordinates:"
                )
                self._hover_idx = idx

            # Always update only the coords while staying in the same box
            coords_only = f"[{int(event.position[0])} {int(event.position[1])}]"
            cur_status = viewer.status
            status = dict(cur_status) if isinstance(cur_status, dict) else {}

            status.update({
                'source_type': 'plugin',     # tell Napari to show the plugin message
                'plugin': self._hover_base,  # unchanged while inside the box
                'layer_base': '',            # blank default layer message
                'help': '',                  # blank help/hints that can override
                'coordinates': coords_only,  #     live coords
            })

            viewer.status = status

    def _update_added_shapes(self, added_items):
        """
        Update the selection box by shape layer names if it they have been added, update current working shape layer and instantiate OrganoiDL if not already there
        """
        # set the latest added shapes layer to the shapes layer that has been selected for saving and visualisation
        self.save_layer_name = added_items[0]
        self.cur_shapes_name = added_items[0]
        self.cur_shapes_layer = self.viewer.layers[self.cur_shapes_name] 

        for layer_name in added_items:
            layer = self.viewer.layers[layer_name]
            self._shape_name_by_id[id(layer)] = layer.name
            # bind rename handler
            layer.events.name.connect(lambda e, layer=layer: self._on_shapes_layer_renamed(layer))

        # self.cur_shapes_layer.events.highlight.connect(self._on_shape_hover)

        # get the bounding box and update the displayed number of organoids
        self._update_num_organoids(len(self.cur_shapes_layer.data)) 
        labels_prop = self.cur_shapes_layer.properties.get(
            'labels',
            [-1] * len(self.cur_shapes_layer.data)
        )        
        # listen for a data change in the current shapes layer
        self.organoiDL.update_bboxes_scores(self.cur_shapes_name,
                                            self.cur_shapes_layer.data,
                                            self.cur_shapes_layer.properties['scores'],
                                            labels_prop,
                                            self.cur_shapes_layer.properties['box_id']
                                            )
        self.cur_shapes_layer.events.data.connect(self.shapes_event_handler)
        self._apply_class_filter()
        self._refresh_class_counts()
        self.organoiDL.update_next_id(self.cur_shapes_name)

        # layer = self.viewer.layers[self.cur_shapes_name]
        # self._shape_name_by_id[id(layer)] = layer.name
        # layer.events.name.connect(lambda e, layer=layer: self._on_shapes_layer_renamed(layer))

        
    # def _update_remove_shapes(self, removed_layers):
    #     """
    #     Update the selection box by removing shape layer names if it they been deleted and set 
    #     """
    #     # update selection box by removing image names if image has been deleted       
    #     for removed_layer in removed_layers:
    #         item_id = self.output_layer_selection.findText(removed_layer)
    #         self.output_layer_selection.removeItem(item_id)
    #         if removed_layer==self.cur_shapes_name: 
    #             self._update_num_organoids(0)
    #             self.organoiDL.remove_shape_from_dict(self.cur_shapes_name)

    #     self._apply_class_filter()
    #     self._refresh_class_counts()

    def _update_remove_shapes(self, removed_layers):
        """
        Update state when shape layers are deleted.
        Clean up backend dicts and rename-tracking state.
        """
        for removed_layer in removed_layers:
            # reset count if this was the active layer
            if removed_layer == self.cur_shapes_name:
                self._update_num_organoids(0)

            # drop backend data for this layer (guard if already gone)
            if removed_layer in self.organoiDL.pred_bboxes:
                self.organoiDL.remove_shape_from_dict(removed_layer)

            # remove any rename-tracking entries that pointed to this name
            dead_ids = [lid for lid, name in list(self._shape_name_by_id.items()) if name == removed_layer]
            for lid in dead_ids:
                self._shape_name_by_id.pop(lid, None)

            # clear pointers if they referenced the removed layer
            if self.cur_shapes_name == removed_layer:
                self.cur_shapes_name = ''
                self.cur_shapes_layer = None
            if self.save_layer_name == removed_layer:
                self.save_layer_name = ''

        self._apply_class_filter()
        self._refresh_class_counts()

    def shapes_event_handler(self, event):
        """
        This function will be called every time the current shapes layer data changes
        """   
        # make sure this stuff isn't done if data in the layer has been changed by the sliders - only by the users
        key = 'napari-organoid-counter:_rerun'
        if key in self.cur_shapes_layer.metadata: 
            return 

        # GUARD: ensure backend dicts exist under the current layer name
        if self.cur_shapes_name not in self.organoiDL.pred_bboxes:
            self.organoiDL.pred_bboxes[self.cur_shapes_name] = torch.empty((0, 4))
            self.organoiDL.pred_scores[self.cur_shapes_name] = torch.empty((0,))
            self.organoiDL.pred_labels[self.cur_shapes_name] = torch.empty((0,), dtype=torch.long)
            self.organoiDL.pred_ids[self.cur_shapes_name] = []
            self.organoiDL.next_id[self.cur_shapes_name] = 1

        # get new ids, new boxes and update the number of organoids
        new_ids = self.viewer.layers[self.cur_shapes_name].properties['box_id']
        self._update_num_organoids(len(new_ids))
        
        # check if duplicate ids - this happens when user adds a box
        if len(new_ids) > len(set(new_ids)):
            num_sim = len(new_ids) - len(set(new_ids))
            if num_sim > 1:  RuntimeWarning('At least one duplicate Box ID found.')
            new_ids[-1] = self.organoiDL.next_id[self.cur_shapes_name]
            new_scores = self.viewer.layers[self.cur_shapes_name].properties['scores']
            new_scores[-1] = 1  # give new box score = 1
    
            # set new properties to shapes layer
            self.viewer.layers[self.cur_shapes_name].properties = {'box_id': new_ids, 'scores': new_scores}
            # refresh text displayed
            self.viewer.layers[self.cur_shapes_name].refresh()
            self.viewer.layers[self.cur_shapes_name].refresh_text()

            labels_prop = self.viewer.layers[self.cur_shapes_name].properties.get(
                'labels', [-1] * len(self.cur_shapes_layer.data)
            )
            self.organoiDL.update_bboxes_scores(
                self.cur_shapes_name,
                self.cur_shapes_layer.data,
                new_scores,
                labels_prop,
                new_ids,
            )

            # and update the OrganoiDL instance
            self.organoiDL.update_next_id(self.cur_shapes_name)
        
        self._apply_class_filter()
        self._refresh_class_counts()


        # this doesn't work!!!!
        # the problem is that the event is called once before the drawing has been completed!!!!!!
        #new_bboxes = self.cur_shapes_layer.data
        #self.organoiDL.update_bboxes_scores(new_bboxes, new_scores, new_ids)
        
    def _setup_input_widget(self):
        """
        Sets up the GUI part which corresposnds to the input configurations
        """
        # setup all the individual boxes
        input_box = self._setup_input_box()
        model_box = self._setup_model_box()
        window_sizes_box = self._setup_window_sizes_box()
        downsampling_box = self._setup_downsampling_box()
        run_box = self._setup_run_box()
        annotation_mode_box = self._setup_annotation_mode_box() # Annotation mode dropdown to select single, multi-annotation or multi-class annotation
        self._setup_progress_box()

        # and add all these to the layout
        input_widget = QGroupBox('Input configurations')
        vbox = QVBoxLayout()
        vbox.addLayout(input_box)
        vbox.addLayout(model_box)
        vbox.addLayout(window_sizes_box)
        vbox.addLayout(downsampling_box)
        vbox.addLayout(run_box)
        vbox.addLayout(annotation_mode_box)  # Add the annotation dropdown
        vbox.addWidget(self.progress_box)
        input_widget.setLayout(vbox)
        return input_widget

    def _setup_output_widget(self):
        """
        Sets up the GUI part which corresposnds to the parameters and outputs
        """
        # setup all the individual boxes
        self.organoid_number_label = QLabel('Number of organoids: '+str(self.num_organoids), self)
        self.organoid_number_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # and add all these to the layout
        self.output_widget = QGroupBox('Parameters and outputs')
        vbox = QVBoxLayout()
        vbox.addLayout(self._setup_min_diameter_box())
        vbox.addLayout(self._setup_confidence_box() )
        if self.annotation_mode != 0:
            # vbox.addWidget(self._setup_color_mapping_box())
            self.legend_box = self._setup_color_mapping_box()  # keep reference
            vbox.addWidget(self.legend_box)
        else:
            self.legend_box = None
        vbox.addWidget(self.organoid_number_label)
        vbox.addLayout(self._setup_reset_box())
        
        self.output_widget.setLayout(vbox)
        return self.output_widget

    def _setup_input_box(self):
        """
        Sets up the GUI part where the input image is defined
        """
        #self.input_box = QGroupBox()
        hbox = QHBoxLayout()
        # setup label
        image_label = QLabel('Image: ', self)
        image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup drop down option for selecting which image to process
        self.image_layer_selection = QComboBox()
        if self.image_layer_names is not None:
            for name in self.image_layer_names: self.image_layer_selection.addItem(name)
        #self.image_layer_selection.setItemText(self.image_layer_name)
        self.image_layer_selection.currentIndexChanged.connect(self._on_image_selection_changed)
        # setup preprocess button to improve visualisation
        preprocess_btn = QPushButton("Preprocess")
        preprocess_btn.clicked.connect(self._on_preprocess_click)
        # and add all these to the layout
        hbox.addWidget(image_label, 2)
        hbox.addWidget(self.image_layer_selection, 4)
        hbox.addWidget(preprocess_btn, 4)
        return hbox

    def _setup_model_box(self):
        """
        Sets up the GUI part where the model is selected from a drop down menu.
        """
        hbox = QHBoxLayout()
        # setup the label
        model_label = QLabel('Model: ', self)
        model_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        # setup the browse files button
        fileOpenButton = QPushButton('Add custom model', self)
        fileOpenButton.show()
        fileOpenButton.clicked.connect(self._on_choose_model_clicked)
        
        # setup drop down option for selecting which image to process
        self.model_selection = QComboBox()
        for name in settings.MODELS.keys(): self.model_selection.addItem(name)
        self.model_selection.setCurrentIndex(self.model_id)
        self.model_selection.currentIndexChanged.connect(self._on_model_selection_changed)
        
        # and add all these to the layout
        hbox.addWidget(model_label, 2)
        hbox.addWidget(self.model_selection, 4)
        hbox.addWidget(fileOpenButton, 4)
        return hbox

    def _setup_window_sizes_box(self):
        """
        Sets up the GUI part where the window sizes parameters are set
        """
        #self.window_sizes_box = QGroupBox()
        hbox = QHBoxLayout()
        info_text = ("Typically a ratio of 512 to 1 between window size and downsampling rate will give good results, (larger window \n"
                    "sizes can lead to a drop in performance). Note that small window sizes will signicantly impact the runtime of the \n"
                    "algorithm. For organoids of different sizes consider setting multiple windows sizes. Hit Enter for the change to \n"
                    "take effect.")
        # setup label
        window_sizes_label = QLabel('Window sizes: [size1, size2, ...]', self)
        window_sizes_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        window_sizes_label.setToolTip(info_text)
        # setup textbox
        self.window_sizes_textbox = QLineEdit(self)
        text = [str(window_size) for window_size in self.window_sizes]
        text = ','.join(text)
        self.window_sizes_textbox.setText(text)
        self.window_sizes_textbox.returnPressed.connect(self._on_window_sizes_changed)
        self.window_sizes_textbox.setToolTip(info_text)
        # and add all these to the layout
        hbox.addWidget(window_sizes_label)
        hbox.addWidget(self.window_sizes_textbox)   
        #self.window_sizes_box.setLayout(hbox)   
        #self.window_sizes_box.setStyleSheet("border: 0px")  
        return hbox
    
    def _setup_downsampling_box(self):
        """
        Sets up the GUI part where the downsampling parameters are set
        """
        #self.downsampling_box = QGroupBox()
        hbox = QHBoxLayout()
        info_text = ("To detect large organoids (and ignore smaller structures) you can increase the downsampling rate. \n"
                    "If your organoids are small and are being missed by the algorithm, consider reducing the downsampling\n"
                    "rate. The number of downsampling inputs should match the number of windows sizes. Hit Enter for the \n"
                    "change to take effect. See window sizes for more info.")

        # setup label
        downsampling_label = QLabel('Downsampling: [ds1, ds2, ...]', self)
        downsampling_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        downsampling_label.setToolTip(info_text)
        # setup textbox
        self.downsampling_textbox = QLineEdit(self)
        text = [str(ds) for ds in self.downsampling]
        text = ','.join(text)
        self.downsampling_textbox.setText(text)
        self.downsampling_textbox.returnPressed.connect(self._on_downsampling_changed)
        self.downsampling_textbox.setToolTip(info_text)
        # and add all these to the layout
        hbox.addWidget(downsampling_label)
        hbox.addWidget(self.downsampling_textbox) 
        #self.downsampling_box.setLayout(hbox)
        #self.downsampling_box.setStyleSheet("border: 0px") 
        return hbox

    def _setup_run_box(self):
        """
        Sets up the GUI part where the user hits the run button
        """
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)
        run_btn.setStyleSheet("border: 0px")
        hbox.addWidget(run_btn)

        # ADD CANCEL BUTTON
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_click)
        self.cancel_btn.setEnabled(False)
        hbox.addWidget(self.cancel_btn)

        hbox.addStretch(1)
        return hbox
    
    def _setup_annotation_mode_box(self):
        """
        Sets up the GUI part where the annotation mode is selected.
        """
        hbox = QHBoxLayout()

        # Label
        annotation_mode_label = QLabel("Number of classes to annotate:", self)
        hbox.addWidget(annotation_mode_label)

        # Dropdown
        self.annotation_mode_dropdown = QComboBox()
        self.annotation_mode_dropdown.addItems(["Detection Only (DO)", "Binary Classification (BC)", "3 classes", "4 classes", "5 classes", "6 classes", "7 classes", "8 classes", "9 classes", "10 classes"])
        self.annotation_mode_dropdown.currentIndexChanged.connect(self.on_annotation_mode_changed)
        hbox.addWidget(self.annotation_mode_dropdown)
        
        return hbox

    def _setup_progress_box(self):
        """
        Sets up the GUI part which appears when the model is being downloaded.
        This should only happen once for each model whihc is then stored in cache. 
        """
        self.progress_box = QGroupBox()
        hbox = QHBoxLayout()
        download_label = QLabel('Downloading model progress: ', self)
        download_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.progress_bar = QProgressBar(self) # creating progress bar
        hbox.addWidget(download_label)
        hbox.addWidget(self.progress_bar)
        self.progress_box.setLayout(hbox)
        self.progress_box.hide()

    def _setup_min_diameter_box(self):
        """
        Sets up the GUI part where the minimum diameter parameter is displayed
        """
        hbox = QHBoxLayout()
        # setup the min diameter slider
        self.min_diameter_slider = QSlider(Qt.Horizontal)
        self.min_diameter_slider.setMinimum(10)
        self.min_diameter_slider.setMaximum(100)
        self.min_diameter_slider.setSingleStep(10)
        self.min_diameter_slider.setValue(self.min_diameter)
        self.min_diameter_slider.valueChanged.connect(self._on_diameter_slider_changed)
        # setup the label
        min_diameter_label = QLabel('Minimum Diameter [um]: ', self)
        min_diameter_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup text box
        self.min_diameter_textbox = QLineEdit(self)
        self.min_diameter_textbox.setText(str(self.min_diameter))
        self.min_diameter_textbox.returnPressed.connect(self._on_diameter_textbox_changed)  
        # and add all these to the layout
        hbox.addWidget(min_diameter_label, 4)
        hbox.addWidget(self.min_diameter_textbox, 1)
        hbox.addWidget(self.min_diameter_slider, 5)
        return hbox

    def _setup_confidence_box(self):
        """
        Sets up the GUI part where the confidence parameter is displayed
        """
        hbox = QHBoxLayout()
        # setup confidence slider
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(5)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setSingleStep(5)
        vis_confidence = int(self.confidence*100)
        self.confidence_slider.setValue(vis_confidence)
        self.confidence_slider.valueChanged.connect(self._on_confidence_slider_changed)
        # setup label
        confidence_label = QLabel('Model confidence: ', self)
        confidence_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup text box
        self.confidence_textbox = QLineEdit(self)
        self.confidence_textbox.setText(str(self.confidence))
        self.confidence_textbox.returnPressed.connect(self._on_confidence_textbox_changed)  
        # and add all these to the layout
        hbox.addWidget(confidence_label, 3)
        hbox.addWidget(self.confidence_textbox, 1)
        hbox.addWidget(self.confidence_slider, 6)
        return hbox

    def _setup_color_mapping_box(self) -> QGroupBox:
        """Build the legend with live counts, per-class check-boxes and a scroll-bar."""
        self.class_count_labels   = {}
        self.class_checkboxes     = {}
        self.visible_classes_filter = set(self.selected_classes)

        outer = QGroupBox("Class-color mapping")
        outer_layout = QVBoxLayout(outer)

        # master “all classes” check-box
        master_row = QHBoxLayout()
        self.master_class_checkbox = QCheckBox("All classes")
        self.master_class_checkbox.setChecked(True)
        self.master_class_checkbox.toggled.connect(self._on_master_class_toggled)
        master_row.addWidget(self.master_class_checkbox)
        master_row.addStretch(1)
        outer_layout.addLayout(master_row)

        # scroll area that will hold the per-class rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # only vertical
        scroll.setFixedHeight(160)          # ≈ 4 rows; tweak if needed

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)

        for cls in sorted(self.selected_classes):
            rgba, name = self.color_mapping[cls]
            r, g, b, a = (int(c * 255) for c in rgba)

            swatch = QLabel()
            swatch.setFixedSize(18, 18)
            swatch.setStyleSheet(
                f"background-color: rgba({r},{g},{b},{a});"
                "border: 1px solid black;"
            )

            label = QLabel()
            self.class_count_labels[cls] = label
            self._update_single_class_label(cls)

            cb = QCheckBox()
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, c=cls: self._on_class_checkbox_toggled(c, checked))
            self.class_checkboxes[cls] = cb

            row = QHBoxLayout()
            row.addWidget(cb)
            row.addWidget(swatch)
            row.addWidget(label)
            row.addStretch(1)
            inner_layout.addLayout(row)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        outer_layout.addWidget(scroll)

        return outer

    def _refresh_color_mapping_box(self):
        """
        Refresh the color mapping box based on the current annotation mode.
        """
        layout = self.output_widget.layout()

        # remove existing legend box from layout (if any)
        if self.legend_box is not None:
            layout.removeWidget(self.legend_box)
            self.legend_box.deleteLater()
            self.legend_box = None

        # Detection-Only → nothing further
        if self.annotation_mode == 0:
            return

        # Build a new one reflecting the current annotation mode
        self.legend_box = self._setup_color_mapping_box()

        self.output_widget.layout().insertWidget(2, self.legend_box)

        self._apply_class_filter()
        self._refresh_class_counts()

    def _update_single_class_label(self, cls: int) -> None:
        """Update the legend row for one class with its current (visible) box-count."""
        if cls not in self.class_count_labels:
            return  # legend not built yet

        if self.cur_shapes_layer is None:
            count = 0
        else:
            colors = np.asarray(self.cur_shapes_layer.edge_color)         # (N,4)
            if colors.size == 0:
                count = 0
            else:
                rgb    = colors[:, :3]
                # alpha  = colors[:, 3]
                target = np.array(self.color_mapping[cls][0][:3])         # (3,)
                mask   = np.all(np.isclose(rgb, target, atol=1e-2), axis=1)  #& (alpha > 1e-3)
                count  = int(mask.sum())

        name = self.color_mapping[cls][1]
        self.class_count_labels[cls].setText(f"Class {cls} ({name}): {count}")

    def _refresh_class_counts(self) -> None:
        """Recompute every class row in the legend."""
        for cls in self.class_count_labels:
            self._update_single_class_label(cls)

    def _current_visible_classes(self) -> set[int]:
        """Return the set of classes currently ticked."""
        # Derive from checkboxes if they exist; otherwise fall back to 'all'
        if not self.class_checkboxes:
            return set(self.selected_classes)
        return {c for c, cb in self.class_checkboxes.items() if cb.isChecked()}

    def _on_master_class_toggled(self, checked: bool) -> None:
        """Master checkbox toggled — set all class checkboxes accordingly."""
        # Prevent signal cascade while we sync children
        for cb in self.class_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

        self.visible_classes_filter = set(self.selected_classes) if checked else set()
        self._apply_class_filter()
        self._refresh_class_counts()

    def _on_class_checkbox_toggled(self, cls: int, checked: bool) -> None:
        """A per-class checkbox toggled."""
        self.visible_classes_filter = self._current_visible_classes()

        # Update master state (checked only if all are checked)
        all_checked = len(self.visible_classes_filter) == len(self.selected_classes)
        if self.master_class_checkbox is not None:
            self.master_class_checkbox.blockSignals(True)
            self.master_class_checkbox.setChecked(all_checked)
            self.master_class_checkbox.blockSignals(False)

        self._apply_class_filter()
        self._refresh_class_counts()

    def _apply_class_filter(self) -> None:
        """Show/hide shapes by class without changing data or recomputing.
        We set edge alpha to 0 for hidden classes and blank their text."""
        if self.annotation_mode == 0:
            return  # Detection-only: no class filtering UI
        if self.cur_shapes_layer is None:
            return

        colors = np.asarray(self.cur_shapes_layer.edge_color).copy()  # (N,4)
        if colors.size == 0:
            return

        # Determine the class of each shape from its edge colour (RGB)
        rgb = colors[:, :3]
        alpha_out = np.ones(len(colors), dtype=float)

        # Hide shapes whose class is NOT selected
        for cls in self.selected_classes:
            target = np.array(self.color_mapping[cls][0][:3])
            matches = np.all(np.isclose(rgb, target, atol=1e-2), axis=1)
            if cls not in self.visible_classes_filter:
                alpha_out[matches] = 0.0  # hide
            else:
                alpha_out[matches] = 1.0  # show

        # Apply new alpha to the edges
        colors[:, 3] = alpha_out
        self.cur_shapes_layer.edge_color = colors

        # Build per-shape text strings (blank for hidden)
        props = self.cur_shapes_layer.properties
        box_ids = props.get('box_id', [])
        scores  = props.get('scores', [])
        # Napari supports list/array of strings for per-shape text
        text_strings = []
        for i in range(len(colors)):
            if alpha_out[i] > 1e-3 and i < len(box_ids) and i < len(scores):
                text_strings.append(f"ID: {int(box_ids[i])}\nConf.: {float(scores[i]):.2f}")
            else:
                text_strings.append("")  # hide text

        # Assign per-shape text
        self.cur_shapes_layer.text = {
            'string': text_strings,
            'size': 9,
            'anchor': 'upper_left',
        }

    def _setup_reset_box(self):
        """
        Sets up the GUI part where screenshot and reset are available to the user
        """
        #self.reset_box = QGroupBox()
        hbox = QHBoxLayout()
        # setup button for resetting parameters
        self.reset_btn = QPushButton("Reset Configs")
        self.reset_btn.clicked.connect(self._on_reset_click)
        # setup button for taking screenshot of current viewer
        self.screenshot_btn = QPushButton("Take screenshot")
        self.screenshot_btn.clicked.connect(self._on_screenshot_click)
        # and add all these to the layout
        hbox.addStretch(1)
        hbox.addWidget(self.screenshot_btn)
        hbox.addSpacing(15)
        hbox.addWidget(self.reset_btn)
        hbox.addStretch(1)
        #self.reset_box.setLayout(hbox)
        #self.reset_box.setStyleSheet("border: 0px")
        return hbox

    def _setup_data_browser_widget(self):
        """
        Sets up the Data Browser section with folder selection, file tree, and navigation buttons.
        """
        data_browser_box = QGroupBox('Data Browser')
        vbox = QVBoxLayout()

        # Folder selection row
        folder_row = QHBoxLayout()
        folder_label = QLabel('Folder:', self)
        self.folder_path_display = QLineEdit(self)
        self.folder_path_display.setReadOnly(True)
        self.folder_path_display.setPlaceholderText('Select a folder...')
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self._on_browse_folder_clicked)
        folder_row.addWidget(folder_label, 1)
        folder_row.addWidget(self.folder_path_display, 6)
        folder_row.addWidget(browse_btn, 2)
        vbox.addLayout(folder_row)

        # File tree widget
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel('Images')
        self.file_tree.setMinimumHeight(200)
        self.file_tree.itemClicked.connect(self._on_tree_item_clicked)
        vbox.addWidget(self.file_tree)

        # Buttons row
        buttons_row = QHBoxLayout()
        buttons_row.addStretch(1)
        self.save_annotation_btn = QPushButton('Save Annotation')
        self.save_annotation_btn.clicked.connect(self._on_save_annotation_clicked)
        self.next_image_btn = QPushButton('Next Image')
        self.next_image_btn.clicked.connect(self._on_next_image_clicked)
        buttons_row.addWidget(self.save_annotation_btn)
        buttons_row.addSpacing(15)
        buttons_row.addWidget(self.next_image_btn)
        buttons_row.addStretch(1)
        vbox.addLayout(buttons_row)

        data_browser_box.setLayout(vbox)
        return data_browser_box

    def _on_browse_folder_clicked(self):
        """Handle folder selection via file dialog."""
        folder = QFileDialog.getExistingDirectory(self, 'Select Data Folder', '')
        if folder:
            self.data_folder = folder
            self.folder_path_display.setText(folder)
            self._scan_folder_for_images(folder)
            self._populate_file_tree()

    def _scan_folder_for_images(self, folder_path: str):
        """Recursively scan folder for supported image files."""
        self.image_files = []
        folder = Path(folder_path)
        for file_path in sorted(folder.rglob('*')):
            if (file_path.is_file()
                    and not file_path.name.startswith('.')
                    and file_path.suffix in self.supported_image_extensions):
                self.image_files.append(file_path)

    def _populate_file_tree(self):
        """Build the tree widget from scanned image files."""
        self.file_tree.clear()
        if not self.data_folder:
            return

        root_path = Path(self.data_folder)
        # Dictionary to store folder items for hierarchy
        folder_items: dict[Path, QTreeWidgetItem] = {}

        for img_path in self.image_files:
            # Get relative path from root
            rel_path = img_path.relative_to(root_path)
            parent_parts = rel_path.parts[:-1]  # All parts except filename

            # Build folder hierarchy
            current_parent = None
            current_path = root_path
            for part in parent_parts:
                current_path = current_path / part
                if current_path not in folder_items:
                    folder_item = QTreeWidgetItem()
                    folder_item.setText(0, part + '/')
                    folder_item.setData(0, Qt.UserRole, None)  # Folders have no path data
                    if current_parent is None:
                        self.file_tree.addTopLevelItem(folder_item)
                    else:
                        current_parent.addChild(folder_item)
                    folder_items[current_path] = folder_item
                current_parent = folder_items[current_path]

            # Add the image file
            file_item = QTreeWidgetItem()
            file_item.setText(0, img_path.name)
            file_item.setData(0, Qt.UserRole, str(img_path))  # Store full path
            if self._is_image_annotated(img_path):
                file_item.setForeground(0, Qt.green)

            if current_parent is None:
                self.file_tree.addTopLevelItem(file_item)
            else:
                current_parent.addChild(file_item)

        # Expand all folders by default
        self.file_tree.expandAll()

    def _is_image_annotated(self, img_path: Path) -> bool:
        """Check if a corresponding JSON annotation file exists."""
        json_path = img_path.with_suffix('.json')
        return json_path.exists()

    def _refresh_file_tree(self):
        """Refresh the file tree to update annotation status indicators."""
        self._populate_file_tree()

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle click on a tree item - load the image if it's a file."""
        img_path_str = item.data(0, Qt.UserRole)
        if img_path_str is None:
            # This is a folder, not a file
            return
        img_path = Path(img_path_str)
        self._load_image_and_annotation(img_path)

    def _auto_save_current(self) -> bool:
        """
        Auto-save the current annotation if there are boxes.
        Returns True if save was performed or no save was needed, False on error.
        """
        if self.current_image_path is None:
            return True  # No image loaded, nothing to save
        
        if self.cur_shapes_layer is None or len(self.cur_shapes_layer.data) == 0:
            return True  # No boxes to save
        
        # Perform save
        return self._save_annotation_for_image(self.current_image_path)

    def _save_annotation_for_image(self, img_path: Path) -> bool:
        """
        Save both JSON and CSV annotation files for the given image.
        Returns True on success, False on failure.
        """
        if self.cur_shapes_layer is None:
            show_info('No shapes layer to save.')
            return False

        bboxes = self.cur_shapes_layer.data
        if len(bboxes) == 0:
            show_info('No organoids to save.')
            return False

        # Get labels from edge colors
        labels, all_valid = self._assign_labels()
        if not all_valid:
            show_error("Some organoids have not been assigned a valid class. Please classify all organoids before saving.")
            return False

        # Prepare file paths
        json_path = img_path.with_suffix('.json')
        csv_path = img_path.with_suffix('.csv')

        # Get bounding box data
        properties = self.cur_shapes_layer.properties
        box_ids = properties.get('box_id', list(range(len(bboxes))))
        scores = properties.get('scores', [1.0] * len(bboxes))
        scale = self.cur_shapes_layer.scale

        # Save JSON (boxes)
        data_json = utils.get_bboxes_as_dict(bboxes, box_ids, scores, scale, labels)
        utils.write_to_json(str(json_path), data_json)

        # Save CSV (features)
        data_csv = utils.get_bbox_diameters(bboxes, box_ids, scale, labels)
        utils.write_to_csv(str(csv_path), data_csv)

        show_info(f'Saved annotation to {json_path.name}')
        return True

    def _load_image_and_annotation(self, img_path: Path):
        """Load an image and its annotation (if exists) into napari."""
        # Auto-save current annotation before loading new image
        self._auto_save_current()

        # Clear existing layers related to the previous image
        layers_to_remove = []
        for layer in self.viewer.layers:
            if isinstance(layer, (layers.Image, layers.Shapes)):
                layers_to_remove.append(layer.name)
        for name in layers_to_remove:
            if name in self.viewer.layers:
                del self.viewer.layers[name]

        # Load the new image
        self.viewer.open(str(img_path))
        self.current_image_path = img_path

        # Update the image layer name in the widget
        new_image_names = self._get_layer_names()
        if new_image_names:
            self.image_layer_name = new_image_names[-1]
            # Convert to grayscale if needed
            img_layer = self.viewer.layers[self.image_layer_name]
            img_data = utils.squeeze_img(img_layer.data)
            if len(img_data.shape) == 3 and img_data.shape[-1] in (3, 4):
                # RGB or RGBA image - convert to grayscale
                from skimage.color import rgb2gray
                gray_data = rgb2gray(img_data[..., :3])
                # Scale to original dtype range
                if img_layer.data.dtype == np.uint8:
                    gray_data = (gray_data * 255).astype(np.uint8)
                elif img_layer.data.dtype == np.uint16:
                    gray_data = (gray_data * 65535).astype(np.uint16)
                img_layer.data = gray_data

        # Check for existing annotation and load it
        json_path = img_path.with_suffix('.json')
        if json_path.exists():
            from napari_organoid_counter._reader import reader_function
            layer_data = reader_function(str(json_path))
            if layer_data:
                for data, attrs, layer_type in layer_data:
                    if layer_type == 'shapes':
                        self.viewer.add_shapes(data, **attrs)

        # Refresh the tree to show updated status
        self._refresh_file_tree()

    def _on_save_annotation_clicked(self):
        """Handle Save Annotation button click."""
        if self.current_image_path is None:
            show_info('No image loaded from Data Browser. Please select an image first.')
            return
        
        if self._save_annotation_for_image(self.current_image_path):
            self._refresh_file_tree()

    def _on_next_image_clicked(self):
        """Find and load the next unannotated image."""
        # First, save current annotation
        if not self._auto_save_current():
            return  # Save failed, don't proceed

        self._refresh_file_tree()

        # Find the next unannotated image
        if not self.image_files:
            show_info('No images in folder. Please select a folder first.')
            return

        # Find current image index
        current_idx = -1
        if self.current_image_path is not None:
            try:
                current_idx = self.image_files.index(self.current_image_path)
            except ValueError:
                current_idx = -1

        # Search for next unannotated image starting from current position
        n = len(self.image_files)
        for i in range(1, n + 1):
            idx = (current_idx + i) % n
            img_path = self.image_files[idx]
            if not self._is_image_annotated(img_path):
                self._load_image_and_annotation(img_path)
                return

        show_info('All images have been annotated!')

    def _get_layer_names(self, layer_type: layers.Layer = layers.Image) -> List[str]:
        """
        Get a list of layer names of a given layer type.
        """
        layer_names = [layer.name for layer in self.viewer.layers if type(layer) == layer_type]
        if layer_names: return [] + layer_names
        else: return []


class ConfirmUpload(QDialog):
    '''
    The QDialog box that appears when the user selects to run organoid counter
    without having the selected model locally
    Parameters
    ----------
        parent: QWidget
            The parent widget, in this case an instance of OrganoidCounterWidget

    '''
    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.setWindowTitle("Confirm Download")
        # setup buttons and text to be displayed
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        text = ("Model not found locally. Downloading default model to \n"
                +str(settings.MODELS_DIR)+"\n"
                "This will only happen once. Click ok to continue or \n"
                "cancel if you do not agree. You won't be able to run\n"
                "the organoid counter if you click cancel.")
        # add all to layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel(text))
        hbox = QHBoxLayout()
        hbox.addWidget(ok_btn)
        hbox.addWidget(cancel_btn)
        layout.addLayout(hbox)
        self.setLayout(layout)
        # connect ok and cancel buttons with accept and reject signals
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
