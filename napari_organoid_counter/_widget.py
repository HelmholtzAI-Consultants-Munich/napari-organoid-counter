import os
from typing import List

from skimage.io import imsave
from datetime import datetime

from napari import layers
from napari.utils.notifications import show_info

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QFileDialog, QLineEdit, QGroupBox)

from napari_organoid_counter._orgacount import OrganoiDL
from napari_organoid_counter._utils import apply_normalization, write_to_csv, get_bbox_diameters, write_to_json, get_bboxes_as_dict, squeeze_img, set_dict_key

import warnings
warnings.filterwarnings("ignore")

class OrganoidCounterWidget(QWidget):
    '''
    The widget of the organoid counter
    Parameters
    ----------
        napari_viewer: string
            The current napari viewer
        model_path: string, default 'model-weights/model_v1.ckpt'
            The relative path to the detection model used for organoid counting - will append current working dir to this path
        window_sizes: list of ints, default [1024]
            A list with the sizes of the windows on which the model will be run. If more than one window_size is given then the model will run on several window sizes and then 
            combne the results
        downsampling:list of ints, default [2]
            A list with the sizes of the downsampling ratios for each window size. List size must be the same as the window_sizes list
        min_diameter: int, default 30
            The minimum organoid diameter given in um
        confidence: float, default 0.8
            The model confidence threhsold - equivalent to box_score_thresh of faster_rcnn
    Attributes
    ----------
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
                model_path: str = 'model/model_v1.ckpt',
                window_sizes: List = [1024],
                downsampling: List = [2],
                min_diameter: int = 30,
                confidence: float = 0.8):
        super().__init__()

        # assign class variables
        self.viewer = napari_viewer 
        self.model_path = os.path.join(os.getcwd(), model_path)
        self.window_sizes = window_sizes
        self.downsampling = downsampling
        self.min_diameter = min_diameter
        self.confidence = confidence

        self.image_layer_names = None
        self.image_layer_name = None 
        self.shape_layer_names = None
        self.save_layer_name = ''
        self.cur_shapes_name = ''
        self.cur_shapes_layer = None
        self.organoiDL = None
        self.num_organoids = 0
        self.original_images = {}
        self.original_contrast = {}
        self.stored_confidences = {}
        self.stored_diameters = {}

        # setup gui        
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._setup_input_widget())
        self.layout().addWidget(self._setup_output_widget())

        # get already opened layers
        self.image_layer_names = self._get_layer_names()
        if len(self.image_layer_names)>0: self._update_added_image(self.image_layer_names)
        self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        if len(self.shape_layer_names)>0: self._update_added_shapes(self.shape_layer_names)
        # and watch for newly added images or shapes
        self.viewer.layers.events.inserted.connect(self._added_layer)
        self.viewer.layers.events.removed.connect(self._removed_layer)
        self.viewer.layers.selection.events.changed.connect(self._sel_layer_changed)

        #self.slider_changed = False # used for changing slider and text of min diameter

    def _sel_layer_changed(self, event):
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
            self.min_diameter_slider.setValue(self.min_diameter)
            #self.min_diameter_label.setText('Minimum Diameter [um]: ')
            self.min_diameter_label.setText('Minimum Diameter [um]: '+str(self.min_diameter))
            # update confidence text and slider with previous value of that layer
            self.confidence = self.stored_confidences[self.cur_shapes_name]
            vis_confidence = int(self.confidence*100)
            self.confidence_slider.setValue(vis_confidence)
            self.confidence_label.setText('Model confidence: '+str(self.confidence))

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
            
    def _removed_layer(self, event):
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
        img = apply_normalization(img)
        self.viewer.layers[self.image_layer_name].data = img
        self.viewer.layers[self.image_layer_name].contrast_limits = (0,255)

    def _update_num_organoids(self, len_bboxes):
        """ Updates the number of organoids displayed in the viewer """
        self.num_organoids = len_bboxes
        new_text = 'Number of organoids: '+str(self.num_organoids)
        self.organoid_number_label.setText(new_text)

    def _update_vis_bboxes(self, bboxes, scores, box_ids, labels_layer_name):
        """ Adds the shapes layer to the viewer or updates it if already there """
        self._update_num_organoids(len(bboxes))
        # if layer already exists
        if labels_layer_name in self.shape_layer_names: 
            self.viewer.layers[labels_layer_name].data = bboxes # hack to get edge_width stay the same!
            self.viewer.layers[labels_layer_name].properties = {'box_id': box_ids,'scores': scores}
            self.viewer.layers[labels_layer_name].edge_width = 12
            self.viewer.layers[labels_layer_name].refresh()
            self.viewer.layers[labels_layer_name].refresh_text()
        # or if this is the first run
        else:
            # if no organoids were found just make an empty shapes layer
            if self.num_organoids==0: 
                self.cur_shapes_layer = self.viewer.add_shapes(name=labels_layer_name)
            # otherwise make the layer and add the boxes
            else:
                properties = {'box_id': box_ids,'scores': scores}
                text_params = {'string': 'ID: {box_id}\nConf.: {scores:.2f}',
                               'size': 12,
                               'anchor': 'upper_left',}
                self.cur_shapes_layer = self.viewer.add_shapes(bboxes, 
                                        name=labels_layer_name,
                                        scale=self.viewer.layers[self.image_layer_name].scale,
                                        face_color='transparent',  
                                        properties = properties,
                                        text = text_params,
                                        edge_color='magenta',
                                        shape_type='rectangle',
                                        edge_width=12) # warning generated here
                # set up event handler for when data from this layer changes
            
            # set current_edge_width so edge width is the same when users annotate - doesnt' fix new preds being added!
            self.viewer.layers[labels_layer_name].current_edge_width = 12


    def _on_preprocess_click(self):
        """ Is called whenever preprocess button is clicked """
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else: self._preprocess()

    def _on_run_click(self):
        """ Is called whenever Run Organoid Counter button is clicked """
        # check if model has been loaded
        if self.organoiDL is None:
            if os.path.isfile(self.model_path):
                self.organoiDL = OrganoiDL(self.viewer.layers[self.image_layer_name].scale,
                                           model_checkpoint=self.model_path
                                           )
            else:
                #show_info('Make sure to select the correct model path!')
                show_info('Model not found locally. Downloading default model instead!')
                self.organoiDL = OrganoiDL(self.viewer.layers[self.image_layer_name].scale,
                                           model_checkpoint=self.model_path
                                           )
                
        # and if an image has been loaded
        if not self.image_layer_name: 
            show_info('Please load an image first and try again!')
            return
        # make sure the number of windows and downsamplings are the same
        if len(self.window_sizes) != len(self.downsampling): 
            show_info('Keep number of window sizes and downsampling the same and try again!')
            return
        # get the current image 
        img_data = self.viewer.layers[self.image_layer_name].data
        # check that image is grayscale
        if len(squeeze_img(img_data).shape) > 2:
            show_info('Only grayscale images currently supported. Try a different image or process it first and try again!')
            return 
        # update the viewer with the new bboxes
        labels_layer_name = 'Labels-'+self.image_layer_name
        if labels_layer_name in self.shape_layer_names:
            show_info('Found existing labels layer. Please remove or rename it and try again!')
            return 
        # run inference
        self.organoiDL.run(img_data, 
                           labels_layer_name,
                           self.window_sizes,
                           self.downsampling,
                           window_overlap = 0.5)
        # set the confidence threshold, remove small organoids and get bboxes in format o visualise
        bboxes, scores, box_ids = self.organoiDL.apply_params(labels_layer_name, self.confidence, self.min_diameter)

        self._update_vis_bboxes(bboxes, scores, box_ids, labels_layer_name)
        # and update cur_shapes_name to newly created shapes layer
        self.cur_shapes_name = labels_layer_name
        # preprocess the image if not done so already to improve visualisation
        self._preprocess() 

    def _on_choose_model_clicked(self):
        """ Is called whenever browse button is clicked for model selection """
        # called when the user hits the 'browse' button to select a model
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.AnyFile)
        if fd.exec_():
            self.model_path = fd.selectedFiles()[0]
        self.model_textbox.setText(self.model_path)
        # initialise organoiDL instance with the model path chosen
        try:
            if self.cur_shapes_layer is not None:
                scale = self.cur_shapes_layer.scale
            elif self.viewer.layers[self.image_layer_name] is not None:
                scale = self.viewer.layers[self.image_layer_name].scale
            else:
                show_info('Could not find a loaded image or annotation file - please load and then select the model')
                return
            self.organoiDL = OrganoiDL(scale,
                                       model_checkpoint=self.model_path, 
                                       )
        except: show_info('Could not load model - make sure you are loading the correct file (with .ckpt ending)')

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
        if self.organoiDL is None:
            self.organoiDL = OrganoiDL(self.cur_shapes_layer.scale,
                                       model_checkpoint=self.model_path)
            self.organoiDL.update_next_id(self.cur_shapes_name, len(self.cur_shapes_layer.scale)+1)

        # make sure to add info to cur_shapes_layer.metadata to differentiate this action from when user adds/removes boxes
        with set_dict_key( self.cur_shapes_layer.metadata, 'napari-organoid-counter:_rerun', True):
            # first update bboxes in organoiDLin case user has added/removed
            self.organoiDL.update_bboxes_scores(self.cur_shapes_name,
                                                self.cur_shapes_layer.data, 
                                                self.cur_shapes_layer.properties['scores'],
                                                self.cur_shapes_layer.properties['box_id'])
            # and get new boxes, scores and box ids based on new confidence and min_diameter values 
            bboxes, scores, box_ids = self.organoiDL.apply_params(self.cur_shapes_name, self.confidence, self.min_diameter)
            self._update_vis_bboxes(bboxes, scores, box_ids, self.cur_shapes_name)
    
    def _on_diameter_changed(self):
        """ Is called whenever user changes the Minimum Diameter slider """
        self.min_diameter = self.min_diameter_slider.value()
        self.min_diameter_label.setText('Minimum Diameter [um]: '+str(self.min_diameter))
        self._rerun()

    '''
    def _on_diameter_slider_changed(self):
        """ Is called whenever user changes the Minimum Diameter slider """
        self.min_diameter = self.min_diameter_slider.value()
        self.slider_changed = True
        if int(self.min_diameter_textbox.text())!= self.min_diameter:
            self.min_diameter_textbox.setText(str(self.min_diameter))
        self._rerun()
        self.slider_changed = False
    
    def _on_diameter_textbox_changed(self):
        if self.slider_changed: return
        self.min_diameter = int(self.min_diameter_textbox.text())
        if self.min_diameter_slider.value() != self.min_diameter:
            self.min_diameter_slider.setValue(self.min_diameter)
        self._rerun()
    '''

    def _on_confidence_changed(self):
        """ Is called whenever user changes the confidence slider """
        self.confidence = self.confidence_slider.value()/100
        self.confidence_label.setText('Model confidence: '+str(self.confidence))
        self._rerun()
        
    def _on_image_selection_changed(self):
        """ Is called whenever a new image has been selected from the drop down box """
        self.image_layer_name = self.image_layer_selection.currentText()
    
    def _on_shapes_selection_changed(self):
        """ Is called whenever a new shapes layer has been selected from the drop down box """
        self.save_layer_name = self.output_layer_selection.currentText()

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

    def _on_save_csv_click(self): 
        """ Is called whenever Save features button is clicked """
        bboxes = self.viewer.layers[self.save_layer_name].data
        if not bboxes: show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
        else:
            # write diameters and area to csv
            data_csv = get_bbox_diameters(bboxes, 
                                          self.viewer.layers[self.save_layer_name].properties['box_id'],
                                          self.viewer.layers[self.save_layer_name].scale)
            fd = QFileDialog()
            name, _ = fd.getSaveFileName(self, 'Save File', self.save_layer_name, 'CSV files (*.csv)')#, 'CSV Files (*.csv)')
            if name: write_to_csv(name, data_csv)


    def _on_save_json_click(self):
        """ Is called whenever Save boxes button is clicked """
        bboxes = self.viewer.layers[self.save_layer_name].data
        #scores = #add
        if not bboxes: show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
        else:
            data_json = get_bboxes_as_dict(bboxes, 
                                           self.viewer.layers[self.save_layer_name].properties['box_id'],
                                           self.viewer.layers[self.save_layer_name].properties['scores'],
                                           self.viewer.layers[self.save_layer_name].scale)
            # write bbox coordinates to json
            fd = QFileDialog()
            name,_ = fd.getSaveFileName(self, 'Save File', self.save_layer_name, 'JSON files (*.json)')#, 'CSV Files (*.csv)')
            if name: write_to_json(name, data_json)

    def _update_added_image(self, added_items):
        """
        Update the selection box with new images if images have been added and update the self.original_images and self.original_contrast dicts.
        Set the latest added image to the current working image (self.image_layer_name)
        """
        for layer_name in added_items:
            self.image_layer_selection.addItem(layer_name)
            self.original_images[layer_name] = self.viewer.layers[layer_name].data
            self.original_contrast[layer_name] = self.viewer.layers[self.image_layer_name].contrast_limits
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

    def _update_added_shapes(self, added_items):
        """
        Update the selection box by shape layer names if it they have been added, update current working shape layer and instantiate OrganoiDL if not already there
        """
        # update the drop down box displaying shape layer names for saving
        for layer_name in added_items:
            self.output_layer_selection.addItem(layer_name)
        # set the latest added shapes layer to the shapes layer that has been selected for saving and visualisation
        self.save_layer_name = added_items[0]
        self.cur_shapes_name = added_items[0]
        self.cur_shapes_layer = self.viewer.layers[self.cur_shapes_name] 
        # get the bounding box and update the displayed number of organoids
        self._update_num_organoids(len(self.cur_shapes_layer.data)) 
        # listen for a data change in the current shapes layer
        self.cur_shapes_layer.events.data.connect(self.shapes_event_handler)
        
    def _update_remove_shapes(self, removed_layers):
        """
        Update the selection box by removing shape layer names if it they been deleted and set 
        """
        # update selection box by removing image names if image has been deleted       
        for removed_layer in removed_layers:
            item_id = self.output_layer_selection.findText(removed_layer)
            self.output_layer_selection.removeItem(item_id)
            if removed_layer==self.cur_shapes_name: 
                self._update_num_organoids(0)
                self.organoiDL = None
                self.cur_shapes_name = '' # DO SOMETHING!

    def shapes_event_handler(self, event):
        """
        This function will be called every time the current shapes layer data changes
        """        
        # make sure this stuff isn't done if data in the layer has been changed by the sliders - only by the users
        key = 'napari-organoid-counter:_rerun'
        if key in self.cur_shapes_layer.metadata: 
            return 
        
        # get new ids, new boxes and update the number of organoids
        new_ids = self.viewer.layers[self.cur_shapes_name].properties['box_id']
        self._update_num_organoids(len(new_ids))
        
        # check if duplicate ids - this happens when user adds a box, currently only available fix current_properties doesn't work
        if len(new_ids) > len(set(new_ids)):
            num_sim = len(new_ids) - len(set(new_ids))
            if num_sim > 1: print('this should not happen!!!!!!!!!!!!!!!!!')
            else: 
                new_ids[-1] = self.organoiDL.next_id[self.cur_shapes_name]
                new_scores = self.viewer.layers[self.cur_shapes_name].properties['scores']
                new_scores[-1] = 1
            # set new properties to shapes layer
            self.viewer.layers[self.cur_shapes_name].properties ={'box_id': new_ids,'scores':  new_scores}
            # refresh text displayed
            self.viewer.layers[self.cur_shapes_name].refresh()
            self.viewer.layers[self.cur_shapes_name].refresh_text()
            # and update the OrganoiDL instance
            self.organoiDL.update_next_id(self.cur_shapes_name)
        
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
        # and add all these to the layout
        input_widget = QGroupBox('Input configurations')
        vbox = QVBoxLayout()
        #vbox.addWidget(self.input_box)
        vbox.addLayout(input_box)
        vbox.addLayout(model_box)
        vbox.addLayout(window_sizes_box)
        vbox.addLayout(downsampling_box)
        vbox.addLayout(run_box)
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
        output_widget = QGroupBox('Parameters and outputs')
        vbox = QVBoxLayout()
        vbox.addLayout(self._setup_min_diameter_box())
        vbox.addLayout(self._setup_confidence_box() )
        vbox.addWidget(self.organoid_number_label)
        vbox.addLayout(self._setup_reset_box())
        vbox.addLayout(self._setup_save_box())
        output_widget.setLayout(vbox)
        return output_widget

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
        hbox.addWidget(image_label)
        hbox.addSpacing(5)
        hbox.addWidget(self.image_layer_selection)
        hbox.addWidget(preprocess_btn)
        #self.input_box.setLayout(hbox)
        #self.input_box.setStyleSheet("border: 0px")
        return hbox

    def _setup_model_box(self):
        """
        Sets up the GUI part where the model path is set
        """
        #self.model_box = QGroupBox()
        hbox = QHBoxLayout()
        # setup the label
        model_label = QLabel('Model: ', self)
        model_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup the model text box
        self.model_textbox = QLineEdit(self)
        self.model_textbox.setText(self.model_path)
        # set up the browse files button
        fileOpenButton = QPushButton('Choose',self)
        fileOpenButton.show()
        fileOpenButton.clicked.connect(self._on_choose_model_clicked)
        # and add all these to the layout
        hbox.addWidget(model_label)
        hbox.addWidget(self.model_textbox)
        hbox.addWidget(fileOpenButton)
        #self.model_box.setLayout(hbox)
        #self.model_box.setStyleSheet("border: 0px")
        return hbox

    def _setup_window_sizes_box(self):
        """
        Sets up the GUI part where the window sizes parameters are set
        """
        #self.window_sizes_box = QGroupBox()
        hbox = QHBoxLayout()
        # setup label
        window_sizes_label = QLabel('Window sizes: [size1, size2, ...]', self)
        window_sizes_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup textbox
        self.window_sizes_textbox = QLineEdit(self)
        text = [str(window_size) for window_size in self.window_sizes]
        text = ','.join(text)
        self.window_sizes_textbox.setText(text)
        self.window_sizes_textbox.returnPressed.connect(self._on_window_sizes_changed)
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
        # setup label
        downsampling_label = QLabel('Downsampling: [ds1, ds2, ...]', self)
        downsampling_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # setup textbox
        self.downsampling_textbox = QLineEdit(self)
        text = [str(ds) for ds in self.downsampling]
        text = ','.join(text)
        self.downsampling_textbox.setText(text)
        self.downsampling_textbox.returnPressed.connect(self._on_downsampling_changed)
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
        hbox.addStretch(1)
        return hbox

    def _setup_min_diameter_box(self):
        """
        Sets up the GUI part where the minimum diameter parameter is displayed
        """
        #self.min_diameter_box = QGroupBox()
        hbox = QHBoxLayout()
        # set up the min diameter slider
        self.min_diameter_slider = QSlider(Qt.Horizontal)
        self.min_diameter_slider.setMinimum(10)
        self.min_diameter_slider.setMaximum(100)
        self.min_diameter_slider.setSingleStep(10)
        self.min_diameter_slider.setValue(self.min_diameter)
        #self.min_diameter_slider.valueChanged.connect(self._on_diameter_slider_changed)
        self.min_diameter_slider.valueChanged.connect(self._on_diameter_changed)
        # set up the label
        #self.min_diameter_label = QLabel('Minimum Diameter [um]: ', self)
        self.min_diameter_label = QLabel('Minimum Diameter [um]: '+str(self.min_diameter), self)
        self.min_diameter_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        '''
        # set up text box
        self.min_diameter_textbox = QLineEdit(self)
        self.min_diameter_textbox.setText(str(self.min_diameter))
        self.min_diameter_textbox.returnPressed.connect(self._on_diameter_textbox_changed)
        '''
        # and add all these to the layout
        hbox.addWidget(self.min_diameter_label)
        #hbox.addWidget(self.min_diameter_textbox)
        hbox.addSpacing(15)
        hbox.addWidget(self.min_diameter_slider)
        #self.min_diameter_box.setLayout(hbox)
        #self.min_diameter_box.setStyleSheet("border: 0px") 
        return hbox

    def _setup_confidence_box(self):
        """
        Sets up the GUI part where the confidence parameter is displayed
        """
        #self.confidence_box = QGroupBox()
        hbox = QHBoxLayout()
        # set up confidence slider
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(5)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setSingleStep(5)
        vis_confidence = int(self.confidence*100)
        self.confidence_slider.setValue(vis_confidence)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        # set up label
        self.confidence_label = QLabel('Model confidence: '+str(self.confidence), self)
        self.confidence_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # and add all these to the layout
        hbox.addWidget(self.confidence_label)
        hbox.addSpacing(15)
        hbox.addWidget(self.confidence_slider)
        #self.confidence_box.setLayout(hbox)
        #self.confidence_box.setStyleSheet("border: 0px") 
        return hbox

    '''
    def _setup_display_res_box(self):

        self.display_res_box = QGroupBox()
        hbox = QHBoxLayout()

        self.organoid_number_label = QLabel('Number of organoids: 0', self)
        self.organoid_number_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
    
        hbox.addWidget(self.organoid_number_label)
        self.display_res_box.setLayout(hbox)
        self.display_res_box.setStyleSheet("border: 0px") 
    '''

    def _setup_reset_box(self):
        """
        Sets up the GUI part where screenshot and reset are available to the user
        """
        #self.reset_box = QGroupBox()
        hbox = QHBoxLayout()
        # set up button for resetting parameters
        self.reset_btn = QPushButton("Reset Configs")
        self.reset_btn.clicked.connect(self._on_reset_click)
        # set up button for taking screenshot of current viewer
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

    def _setup_save_box(self):
        """
        Sets up the GUI part where shapes layer is saved 
        """
        #self.save_box = QGroupBox()
        hbox = QHBoxLayout()
        # set up button for saving features
        self.save_csv_btn = QPushButton("Save features")
        self.save_csv_btn.clicked.connect(self._on_save_csv_click)
        # set up button for saving boxes
        self.save_json_btn = QPushButton("Save boxes")
        self.save_json_btn.clicked.connect(self._on_save_json_click)
        # set up label
        self.save_label = QLabel('Save: ', self)
        self.save_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        # set up drop down option for selecting which shapes layer to save
        self.output_layer_selection = QComboBox()
        if self.shape_layer_names is not None:
            for name in self.shape_layer_names: self.output_layer_selection.addItem(name)
        self.output_layer_selection.currentIndexChanged.connect(self._on_shapes_selection_changed)
        # and add all these to the layout
        hbox.addWidget(self.save_label)
        hbox.addSpacing(5)
        hbox.addWidget(self.output_layer_selection)
        hbox.addWidget(self.save_csv_btn)
        hbox.addWidget(self.save_json_btn)
        #self.save_box.setLayout(hbox)
        #self.save_box.setStyleSheet("border: 0px")
        return hbox

    def _get_layer_names(self, layer_type: layers.Layer = layers.Image) -> List[str]:
        """
        Get a list of layer names of a given layer type.
        """
        layer_names = [layer.name for layer in self.viewer.layers if type(layer) == layer_type]
        if layer_names: return [] + layer_names
        else: return []

