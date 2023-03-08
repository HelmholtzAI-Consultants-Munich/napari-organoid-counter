import os
from typing import List

from skimage.io import imsave
from datetime import datetime

from napari import layers
from napari.utils.notifications import show_info

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QFileDialog, QLineEdit, QGroupBox)

from napari_organoid_counter._orgacount import OrganoiDL
from napari_organoid_counter._utils import apply_normalization, write_to_csv, get_bbox_diameters, write_to_json, get_bboxes_as_dict, squeeze_img

import warnings
warnings.filterwarnings("ignore")

class OrganoidCounterWidget(QWidget):
    # the widget of the organoid counter - documentation to be added
    def __init__(self, 
                napari_viewer,
                model_path: str = 'model-weights/model_v1.ckpt',
                window_sizes: List = [2048],
                downsampling: List = [2],
                min_diameter: int = 30,
                confidence: float = 0.8):
        super().__init__()
        
        # assign class variables
        self.viewer = napari_viewer
        self.organoiDL = None
        self.cur_shapes = None
        self.model_path = os.path.join(os.getcwd(), model_path)
        self.window_sizes = window_sizes
        self.downsampling = downsampling
        self.min_diameter = min_diameter
        self.confidence = confidence
        self.num_organoids = 0
        
        # read already opened files
        self.image_layer_names = self._get_layer_names()
        self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
        self.original_images = {}
        self.original_labels = {}
        self.original_contrast = {}
        if self.image_layer_names: 
            self.image_layer_name = self.image_layer_names[0]
            for layer_name in self.image_layer_names:
                self.original_images[layer_name] = self.viewer.layers[layer_name].data
                self.original_contrast[layer_name] = self.viewer.layers[layer_name].contrast_limits
        else: self.image_layer_name = None
        if self.shape_layer_names: self.self.shape_layer_name = self.shape_layer_names[0]

        # setup gui
        self._setup_input_widget()
        self._setup_output_widget()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.input_widget)
        self.layout().addWidget(self.output_widget)
        
        @self.viewer.layers.events.connect
        def _added_layer(arg): 

            # get image layers names
            self.image_layer_names = self._get_layer_names()
            current_selection_items = [self.image_layer_selection.itemText(i) for i in range(self.image_layer_selection.count())]
            self._update_added_image(current_selection_items)
            self._update_removed_image(current_selection_items)

            # do the same with shapes layers
            self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
            current_selection_items = [self.output_layer_selection.itemText(i) for i in range(self.output_layer_selection.count())]
            self._update_added_shapes(current_selection_items)
            self._update_remove_shapes(current_selection_items)

    def _preprocess(self):
        """Preprocess the current image in the viewer to improve visualisation for the user"""
        img = self.original_images[self.image_layer_name]
        img = apply_normalization(img)
        self.viewer.layers[self.image_layer_name].data = img
        self.viewer.layers[self.image_layer_name].contrast_limits = (0,255)

    def _update_num_organoids(self, bboxes):
        """Updates the number of organoids displayed in the viewer"""
        self.num_organoids = len(bboxes)
        new_text = 'Number of organoids: '+str(self.num_organoids)
        self.organoid_number_label.setText(new_text)

    def _update_vis_bboxes(self, bboxes, scores):
        """ Adds the shapes layer to the viewer or updates it if already there"""
        self._update_num_organoids(bboxes)
        seg_layer_name = 'Labels-'+self.image_layer_name

        if seg_layer_name in self.shape_layer_names: 
            self.cur_shapes_layer.add(bboxes, 
                                      face_color='transparent',  
                                      edge_color='magenta',
                                      shape_type='rectangle',
                                      edge_width=12)
            self.viewer.layers[seg_layer_name].data = bboxes # hack to get edge_width stay the same!
            self.viewer.layers[seg_layer_name].current_properties = {'box_id': self.num_organoids+1,'scores':  1}

        else:
            # if no organoids were found just make an empty shapes layer
            if self.num_organoids==0: 
                self.cur_shapes_layer = self.viewer.add_shapes(name=seg_layer_name)
            # otherwise make the layer and add the boxes
            else:
                properties = {'box_id': list(range(self.num_organoids)),'scores': scores}
                text_params = {'string': 'ID: {box_id}\nConf.: {scores:.2f}',
                               'size': 12,
                               'anchor': 'upper_left',}
                self.cur_shapes_layer = self.viewer.add_shapes(bboxes, 
                                        name=seg_layer_name,
                                        scale=self.viewer.layers[self.image_layer_name].scale,
                                        face_color='transparent',  
                                        properties = properties,
                                        text = text_params,
                                        edge_color='magenta',
                                        shape_type='rectangle',
                                        edge_width=12) # warning generated here

        self.cur_shapes_layer.events.data.connect(self.shapes_event_handler)
        self.viewer.layers[seg_layer_name].current_properties = {'box_id': self.num_organoids+1,'scores':  1}
        self.viewer.layers[seg_layer_name].current_edge_width = 12 # so edge width is the same when users annotate - doesnt' fix new preds being added!
        self.cur_shapes = seg_layer_name

    def _on_preprocess_click(self):
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else: self._preprocess()

    def _on_run_click(self):
        # check if model has been loaded
        if self.organoiDL is None:
            if os.path.isfile(self.model_path):
                self.organoiDL = OrganoiDL(model_checkpoint=self.model_path)
            else:
                show_info('Make sure to select the correct model path!')
                return
        # and if an image has been loaded
        if not self.image_layer_name: 
            show_info('Please load an image first and try again!')
            return
        # make sure the number of windows and downsamplings are the same
        if len(self.window_sizes) != len(self.downsampling): 
            show_info('Keep number of window sizes and downsampling the same and try again!')
            return
        # get the current image and scle in um
        img_data = self.viewer.layers[self.image_layer_name].data
        img_scale = self.viewer.layers[self.image_layer_name].scale
        # check that image is grayscale
        if len(squeeze_img(img_data).shape) > 2:
            show_info('Only grayscale images currently supported. Try a different image or process it first and try again!')
            return
        # run inference
        self.organoiDL.run(img_data, 
                           img_scale,
                           self.window_sizes,
                           self.downsampling,
                           window_overlap = 1)# 0.5)
        # set the confidence threshold, remove small organoids and get bboxes in format o visualise
        bboxes, scores = self.organoiDL.apply_params(self.confidence, self.min_diameter)
        # update the viewer with the new bboxes
        self._update_vis_bboxes(bboxes, scores)
        # preprocess the image if not done so already to improve visualisation
        self._preprocess() 

    def _on_choose_model_clicked(self):
        # called when the user hits the 'browse' button to select a model
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.AnyFile)
        if fd.exec_():
            self.model_path = fd.selectedFiles()[0]
        self.model_textbox.setText(self.model_path)
        # initialise organoiDL instance with the model path chosen
        try:
            self.organoiDL = OrganoiDL(model_checkpoint=self.model_path)
        except: show_info('Could not load model - make sure you are loading the correct file (with .ckpt ending)')

    def _on_window_sizes_changed(self):
        new_window_sizes = self.window_sizes_textbox.text()
        new_window_sizes = new_window_sizes.split(',')
        self.window_sizes = [int(win_size) for win_size in new_window_sizes]

    def _on_downsampling_changed(self):
        new_downsampling = self.downsampling_textbox.text()
        new_downsampling = new_downsampling.split(',')
        self.downsampling = [int(ds) for ds in new_downsampling]

    def _rerun(self):
        if self.organoiDL is not None:
            bboxes, scores = self.organoiDL.apply_params(self.confidence, self.min_diameter)
            self._update_vis_bboxes(bboxes, scores)

    def _on_diameter_changed(self):
        self.min_diameter = self.min_diameter_slider.value()
        self.min_diameter_label.setText('Minimum Diameter [um]: '+str(self.min_diameter))
        self._rerun()

    def _on_confidence_changed(self):
        self.confidence = self.confidence_slider.value()/100
        self.confidence_label.setText('Model confidence: '+str(self.confidence))
        self._rerun()
        
    def _on_image_selection_changed(self):
        self.image_layer_name = self.image_layer_selection.currentText()
    
    def _on_shapes_selection_changed(self):
        self.shape_layer_name = self.output_layer_selection.currentText()
    '''
    def _on_update_click(self):
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else:
            #selected_layer_name = self.output_layer_selection.currentText()
            bboxes = self.viewer.layers[self.cur_shapes].data
            self._update_num_organoids(bboxes)
            self.num_organoids = len(bboxes)
            new_text = 'Number of organoids: '+str(self.num_organoids)
            self.organoid_number_label.setText(new_text)
        
            #self._preprocess()
            #img_data = self.viewer.layers[self.image_layer_name].data # get pre-processed image!!!
            #bboxes = self.viewer.layers['Labels-'+self.image_layer_name].data
            #img_data = add_text_to_img(img_data, len(bboxes))
            #self.viewer.layers[self.image_layer_name].data = img_data
        
    '''
    def _on_reset_click(self):
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
        screenshot=self.viewer.screenshot()
        if not self.image_layer_name: potential_name = datetime.now().strftime("%d%m%Y%H%M%S")+'screenshot.png'
        else: potential_name = self.image_layer_name+datetime.now().strftime("%d%m%Y%H%M%S")+'_screenshot.png'
        fd = QFileDialog()
        name,_ = fd.getSaveFileName(self, 'Save File', potential_name, 'Image files (*.png);;(*.tiff)') #, 'CSV Files (*.csv)')
        if name: imsave(name, screenshot)

    def _on_save_csv_click(self): 
        selected_layer_name = self.output_layer_selection.currentText()
        bboxes = self.viewer.layers[selected_layer_name].data
        if not bboxes: show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
        else:
            # write diameters and area to csv
            data_csv = get_bbox_diameters(bboxes, 
                                          self.viewer.layers[self.shape_layer_name].properties.box_id,
                                          self.viewer.layers[self.shape_layer_name].scale)
            fd = QFileDialog()
            name, _ = fd.getSaveFileName(self, 'Save File', self.shape_layer_name, 'CSV files (*.csv)')#, 'CSV Files (*.csv)')
            if name: write_to_csv(name, data_csv)


    def _on_save_json_click(self):
        selected_layer_name = self.output_layer_selection.currentText()
        bboxes = self.viewer.layers[selected_layer_name].data
        #scores = #add
        if not bboxes: show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
        else:
            data_json = get_bboxes_as_dict(bboxes, 
                                           self.viewer.layers[self.shape_layer_name].properties.box_id,
                                           self.viewer.layers[self.shape_layer_name].properties.scores,
                                           self.viewer.layers[self.shape_layer_name].scale)
            # write bbox coordinates to json
            fd = QFileDialog()
            name,_ = fd.getSaveFileName(self, 'Save File', self.shape_layer_name, 'JSON files (*.json)')#, 'CSV Files (*.csv)')
            if name: write_to_json(name, data_json)


    def _setup_input_widget(self):

        input_box = self._setup_input_box()
        model_box = self._setup_model_box()
        window_sizes_box = self._setup_window_sizes_box()
        downsampling_box = self._setup_downsampling_box()
        run_box = self._setup_run_box()

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)
        run_btn.setStyleSheet("border: 0px")
        hbox.addWidget(run_btn)
        hbox.addStretch(1)
        
        self.input_widget = QGroupBox('Input configurations')
        vbox = QVBoxLayout()
        #vbox.addWidget(self.input_box)
        vbox.addLayout(input_box)
        vbox.addLayout(model_box)
        vbox.addLayout(window_sizes_box)
        vbox.addLayout(downsampling_box)
        vbox.addLayout(run_box)
        self.input_widget.setLayout(vbox)

    def _setup_output_widget(self):

        self.organoid_number_label = QLabel('Number of organoids: '+str(self.num_organoids), self)
        self.organoid_number_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.output_widget = QGroupBox('Parameters and outputs')
        vbox = QVBoxLayout()
        vbox.addLayout(self._setup_min_diameter_box())
        vbox.addLayout(self._setup_confidence_box() )
        vbox.addWidget(self.organoid_number_label)
        vbox.addLayout(self._setup_reset_box())
        vbox.addLayout(self._setup_save_box())
        self.output_widget.setLayout(vbox)


    def _setup_input_box(self):

        #self.input_box = QGroupBox()
        hbox = QHBoxLayout()

        image_label = QLabel('Image: ', self)
        image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.image_layer_selection = QComboBox()
        for name in self.image_layer_names:
            self.image_layer_selection.addItem(name)
        #self.image_layer_selection.setItemText(self.image_layer_name)
        self.image_layer_selection.currentIndexChanged.connect(self._on_image_selection_changed)
    
        preprocess_btn = QPushButton("Preprocess")
        preprocess_btn.clicked.connect(self._on_preprocess_click)

        hbox.addWidget(image_label)
        hbox.addSpacing(5)
        hbox.addWidget(self.image_layer_selection)
        hbox.addWidget(preprocess_btn)
        #self.input_box.setLayout(hbox)
        #self.input_box.setStyleSheet("border: 0px")
        return hbox

    def _setup_model_box(self):

        #self.model_box = QGroupBox()
        hbox = QHBoxLayout()
        
        model_label = QLabel('Model: ', self)
        model_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        
        self.model_textbox = QLineEdit(self)
        self.model_textbox.setText(self.model_path)
        
        fileOpenButton = QPushButton('Choose',self)
        fileOpenButton.show()
        fileOpenButton.clicked.connect(self._on_choose_model_clicked)

        hbox.addWidget(model_label)
        hbox.addWidget(self.model_textbox)
        hbox.addWidget(fileOpenButton)
        #self.model_box.setLayout(hbox)
        #self.model_box.setStyleSheet("border: 0px")
        return hbox

    def _setup_window_sizes_box(self):

        #self.window_sizes_box = QGroupBox()
        hbox = QHBoxLayout()

        window_sizes_label = QLabel('Window sizes: [size1, size2, ...]', self)
        window_sizes_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.window_sizes_textbox = QLineEdit(self)
        text = [str(window_size) for window_size in self.window_sizes]
        text = ','.join(text)
        self.window_sizes_textbox.setText(text)
        self.window_sizes_textbox.returnPressed.connect(self._on_window_sizes_changed)

        hbox.addWidget(window_sizes_label)
        hbox.addWidget(self.window_sizes_textbox)   
        #self.window_sizes_box.setLayout(hbox)   
        #self.window_sizes_box.setStyleSheet("border: 0px")  
        return hbox


    def _setup_downsampling_box(self):

        #self.downsampling_box = QGroupBox()
        hbox = QHBoxLayout()

        downsampling_label = QLabel('Downsampling: [ds1, ds2, ...]', self)
        downsampling_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.downsampling_textbox = QLineEdit(self)
        text = [str(ds) for ds in self.downsampling]
        text = ','.join(text)
        self.downsampling_textbox.setText(text)
        self.downsampling_textbox.returnPressed.connect(self._on_downsampling_changed)

        hbox.addWidget(downsampling_label)
        hbox.addWidget(self.downsampling_textbox) 
        #self.downsampling_box.setLayout(hbox)
        #self.downsampling_box.setStyleSheet("border: 0px") 
        return hbox

    def _setup_run_box(self):
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)
        run_btn.setStyleSheet("border: 0px")
        hbox.addWidget(run_btn)
        hbox.addStretch(1)
        return hbox

    def _setup_min_diameter_box(self):

        #self.min_diameter_box = QGroupBox()
        hbox = QHBoxLayout()

        self.min_diameter_slider = QSlider(Qt.Horizontal)
        self.min_diameter_slider.setMinimum(10)
        self.min_diameter_slider.setMaximum(100)
        self.min_diameter_slider.setSingleStep(10)
        self.min_diameter_slider.setValue(self.min_diameter)
        self.min_diameter_slider.valueChanged.connect(self._on_diameter_changed)

        self.min_diameter_label = QLabel('Minimum Diameter [um]: '+str(self.min_diameter), self)
        self.min_diameter_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        #self.min_diameter_label.setMinimumWidth(80)

        hbox.addWidget(self.min_diameter_label)
        hbox.addSpacing(15)
        hbox.addWidget(self.min_diameter_slider)
        #self.min_diameter_box.setLayout(hbox)
        #self.min_diameter_box.setStyleSheet("border: 0px") 
        return hbox

    def _setup_confidence_box(self):

        #self.confidence_box = QGroupBox()
        hbox = QHBoxLayout()

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(5)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setSingleStep(5)
        vis_confidence = int(self.confidence*100)
        self.confidence_slider.setValue(vis_confidence)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)

        self.confidence_label = QLabel('Model confidence: '+str(self.confidence), self)
        self.confidence_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

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
        #self.reset_box = QGroupBox()
        hbox = QHBoxLayout()

        self.reset_btn = QPushButton("Reset Configs")
        self.reset_btn.clicked.connect(self._on_reset_click)

        self.screenshot_btn = QPushButton("Take screenshot")
        self.screenshot_btn.clicked.connect(self._on_screenshot_click)
       
        hbox.addStretch(1)
        hbox.addWidget(self.screenshot_btn)
        hbox.addSpacing(15)
        hbox.addWidget(self.reset_btn)
        hbox.addStretch(1)
        #self.reset_box.setLayout(hbox)
        #self.reset_box.setStyleSheet("border: 0px")
        return hbox

    def _setup_save_box(self):
        
        #self.save_box = QGroupBox()
        hbox = QHBoxLayout()

        self.save_csv_btn = QPushButton("Save features")
        self.save_csv_btn.clicked.connect(self._on_save_csv_click)

        self.save_json_btn = QPushButton("Save boxes")
        self.save_json_btn.clicked.connect(self._on_save_json_click)

        self.save_label = QLabel('Save: ', self)
        self.save_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.output_layer_selection = QComboBox()
        for name in self.shape_layer_names:
            self.output_layer_selection.addItem(name)
        self.output_layer_selection.currentIndexChanged.connect(self._on_shapes_selection_changed)
    

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
        Get list of layer names of a given layer type.
        """
        layer_names = [layer.name for layer in self.viewer.layers if type(layer) == layer_type]

        if layer_names:
            return [] + layer_names
        else:
            return []

    def _update_added_image(self, current_selection_items):
        # update selection box with new images if image has been added
        for layer_name in self.image_layer_names:
            if layer_name not in current_selection_items: 
                self.image_layer_selection.addItem(layer_name)
                self.original_images[layer_name] = self.viewer.layers[layer_name].data
                self.original_contrast[layer_name] = self.viewer.layers[self.image_layer_name].contrast_limits


    def _update_removed_image(self, current_selection_items):
        # update selection box by removing image names if image has been deleted       
        removed_layers = [name for name in current_selection_items if name not in self.image_layer_names]
        for removed_layer in removed_layers:
            item_id = self.image_layer_selection.findText(removed_layer)
            self.image_layer_selection.removeItem(item_id)
            del self.original_images[removed_layer]
            del self.original_contrast[removed_layer]

    def _update_added_shapes(self, current_selection_items):
        for layer_name in self.shape_layer_names:
            if layer_name not in current_selection_items: 
                self.output_layer_selection.addItem(layer_name)

    def _update_remove_shapes(self, current_selection_items):
        # update selection box by removing image names if image has been deleted       
        removed_layers = [name for name in current_selection_items if name not in self.shape_layer_names]
        for removed_layer in removed_layers:
            item_id = self.output_layer_selection.findText(removed_layer)
            self.output_layer_selection.removeItem(item_id)

    def shapes_event_handler(self, event):
        self._update_num_organoids(self.cur_shapes_layer.data)
