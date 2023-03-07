import os
import csv
import json
from skimage.io import imsave
from datetime import datetime
from typing import List

from napari import layers
#from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QFileDialog, QLineEdit, QGroupBox)
from napari.utils.notifications import show_info
import math
from napari_organoid_counter._orgacount import OrganoiDL
from napari_organoid_counter._utils import apply_normalization

import warnings
warnings.filterwarnings("ignore")

class OrganoidCounterWidget(QWidget):
    # the widget of the organoid counter - documentation to be added
    def __init__(self, 
                napari_viewer,
                model_path: str = 'model-weights/model_eva_v0.ckpt',
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
        
        # this has to be changed if we later add more images it needs to be updated 
        self.image_layer_names = self._get_layer_names()
        self.original_images = {}
        self.original_contrast = {}
        if self.image_layer_names: 
            self.image_layer_name = self.image_layer_names[0]
            for layer_name in self.image_layer_names:
                self.original_images[layer_name] = self.viewer.layers[layer_name].data
                self.original_contrast[layer_name] = self.viewer.layers[self.image_layer_name].contrast_limits
        else: self.image_layer_name = None
        
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
            # update selection box with new images if image has been added
            current_selection_items = [self.image_layer_selection.itemText(i) for i in range(self.image_layer_selection.count())]
            for layer_name in self.image_layer_names:
                if layer_name not in current_selection_items: 
                    self.image_layer_selection.addItem(layer_name)
                    self.original_images[layer_name] = self.viewer.layers[layer_name].data
                    self.original_contrast[layer_name] = self.viewer.layers[self.image_layer_name].contrast_limits
            # update selection box by removing image names if image has been deleted       
            removed_layers = [name for name in current_selection_items if name not in self.image_layer_names]
            for removed_layer in removed_layers:
                item_id = self.image_layer_selection.findText(removed_layer)
                self.image_layer_selection.removeItem(item_id)
                del self.original_images[removed_layer]
                del self.original_contrast[removed_layer]
            # do the same with shapes layers
            self.shape_layer_names = self._get_layer_names(layer_type=layers.Shapes)
            current_selection_items = [self.output_layer_selection.itemText(i) for i in range(self.output_layer_selection.count())]
            for layer_name in self.shape_layer_names:
                if layer_name not in current_selection_items: 
                    self.output_layer_selection.addItem(layer_name)
            # update selection box by removing image names if image has been deleted       
            removed_layers = [name for name in current_selection_items if name not in self.shape_layer_names]
            for removed_layer in removed_layers:
                item_id = self.output_layer_selection.findText(removed_layer)
                self.output_layer_selection.removeItem(item_id)

    def _preprocess(self):
        img = self.original_images[self.image_layer_name]
        img = apply_normalization(img)
        self.viewer.layers[self.image_layer_name].data = img
        self.viewer.layers[self.image_layer_name].contrast_limits = (0,255)

    def _update_vis_bboxes(self, bboxes):
        new_text = 'Number of detected organoids: '+str(len(bboxes))
        self.organoid_number_label.setText(new_text)
        seg_layer_name = 'Organoids '+self.image_layer_name
        if seg_layer_name in self.shape_layer_names: 
            
            self.cur_shapes_layer.add(bboxes, 
                                      face_color='transparent',  
                                      edge_color='magenta',
                                      shape_type='rectangle',
                                      edge_width=12)
            self.viewer.layers[seg_layer_name].data = bboxes
            '''
            self.viewer.layers[seg_layer_name].current_edge_width = 12
            #self.viewer.layers[seg_layer_name].edge_width[:] = 12 #TypeError: can only assign an iterable
            self.viewer.layers[seg_layer_name].refresh()
            print(self.viewer.layers[seg_layer_name].edge_width)
            '''

        else:
            if len(bboxes)==0: 
                self.cur_shapes_layer = self.viewer.add_shapes(name = seg_layer_name)
            else:
                self.cur_shapes_layer = self.viewer.add_shapes(bboxes, 
                                        name=seg_layer_name,
                                        scale=self.viewer.layers[self.image_layer_name].scale,
                                        face_color='transparent',  
                                        edge_color='magenta',
                                        shape_type='rectangle',
                                        edge_width=12) # warning generated here
        self.viewer.layers[seg_layer_name].current_edge_width = 12
        self.cur_shapes = seg_layer_name

    def _on_preprocess_click(self):
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else: self._preprocess()

    def _on_run_click(self):

        if self.organoiDL is None:
            if os.path.isfile(self.model_path):
                self.organoiDL = OrganoiDL(model_checkpoint=self.model_path)
            else:
                show_info('Make sure to select the correct model path!')
                return
        
        if not self.image_layer_name: 
            show_info('Please load an image first and try again!')
            return

        if len(self.window_sizes) != len(self.downsampling): 
            show_info('Keep number of window sizes and downsampling the same and try again!')
            return

        img_data = self.viewer.layers[self.image_layer_name].data
        img_scale = self.viewer.layers[self.image_layer_name].scale
        
        self.organoiDL.run(img_data, 
                           img_scale,
                           self.window_sizes,
                           self.downsampling,
                           window_overlap = 1)# 0.5)
        bboxes = self.organoiDL.apply_params(self.confidence, self.min_diameter)
        
        self._preprocess() # preprocess if not done so already to improve visualisation
        self._update_vis_bboxes(bboxes)

    def _on_choose_model_clicked(self):
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.AnyFile)
        if fd.exec_():
            self.model_path = fd.selectedFiles()[0]
        self.model_textbox.setText(self.model_path)
        self.organoiDL = OrganoiDL(model_checkpoint=self.model_path)

    def _on_window_sizes_changed(self):
        new_window_sizes = self.window_sizes_textbox.text()
        new_window_sizes = new_window_sizes.split(',')
        self.window_sizes = [int(win_size) for win_size in new_window_sizes]

    def _on_downsampling_changed(self):
        new_downsampling = self.downsampling_textbox.text()
        new_downsampling = new_downsampling.split(',')
        self.downsampling = [int(ds) for ds in new_downsampling]

    def _on_diameter_changed(self):
        self.min_diameter = self.min_diameter_slider.value()
        self.min_diameter_label.setText('Minimum Diameter [um]: '+str(self.min_diameter))
        if self.organoiDL is not None:
            bboxes = self.organoiDL.apply_params(self.confidence, self.min_diameter)
            self._update_vis_bboxes(bboxes)

    def _on_confidence_changed(self):
        self.confidence = self.confidence_slider.value()/100
        self.confidence_label.setText('Model confidence: '+str(self.confidence))
        if self.organoiDL is not None:
            bboxes = self.organoiDL.apply_params(self.confidence, self.min_diameter)
            self._update_vis_bboxes(bboxes)
        
    def _image_selection_changed(self):
        self.image_layer_name = self.image_layer_selection.currentText()

    def _on_update_click(self):
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else:
            #selected_layer_name = self.output_layer_selection.currentText()
            bboxes = self.viewer.layers[self.cur_shapes].data
            new_text = 'Number of detected organoids: '+str(len(bboxes))
            self.organoid_number_label.setText(new_text)
        '''
            self._preprocess()
            img_data = self.viewer.layers[self.image_layer_name].data # get pre-processed image!!!
            bboxes = self.viewer.layers['Organoids '+self.image_layer_name].data
            img_data = add_text_to_img(img_data, len(bboxes))
            self.viewer.layers[self.image_layer_name].data = img_data
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
            data_csv = []
            # save diameters and area of organoids (approximated as ellipses)
            for i, bbox in enumerate(bboxes):
                d1 = abs(bbox[0][0] - bbox[2][0]) * self.viewer.layers[self.image_layer_name].scale[0]
                d2 = abs(bbox[0][1] - bbox[2][1]) * self.viewer.layers[self.image_layer_name].scale[0]
                area = math.pi * d1 * d2
                data_csv.append([i, round(d1,3), round(d2,3), round(area,3)])
            # write diameters and area to csv
            potential_name = self.image_layer_name + '_features'
            fd = QFileDialog()
            name,_ = fd.getSaveFileName(self, 'Save File', potential_name, 'CSV files (*.csv)')#, 'CSV Files (*.csv)')
            if name:
                with open(name, 'w') as f:
                    write = csv.writer(f, delimiter=';')
                    write.writerow(['OrganoidID', 'D1[um]','D2[um]', 'Area [um^2]'])
                    write.writerows(data_csv)

    def _on_save_json_click(self):
        selected_layer_name = self.output_layer_selection.currentText()
        bboxes = self.viewer.layers[selected_layer_name].data
        if not bboxes: show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
        else:
            data_json = {} 
            for i, bbox in enumerate(bboxes):
                data_json.update({str(i): [list(bboxit) for bboxit in bbox]})
            # write bbox coordinates to json
            potential_name = self.image_layer_name + '_annotations'
            fd = QFileDialog()
            name,_ = fd.getSaveFileName(self, 'Save File', potential_name, 'JSON files (*.json)')#, 'CSV Files (*.csv)')
            if name:
                with open(name, 'w') as outfile:
                    json.dump(data_json, outfile)  

    def _setup_input_widget(self):

        self._setup_input_box()
        self._setup_model_box()
        self._setup_window_sizes_box()
        self._setup_downsampling_box()

        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)
        run_btn.setStyleSheet("border: 0px")
        
        self.input_widget = QGroupBox('Input configurations')
        vbox = QVBoxLayout()
        vbox.addWidget(self.input_box)
        vbox.addWidget(self.model_box)
        vbox.addWidget(self.window_sizes_box)
        vbox.addWidget(self.downsampling_box)
        vbox.addWidget(run_btn)
        self.input_widget.setLayout(vbox)

    def _setup_output_widget(self):
        
        self._setup_min_diameter_box()
        self._setup_confidence_box()        
        self._setup_display_res_box()
        self._setup_reset_box()
        self._setup_save_box()

        self.output_widget = QGroupBox('Parameters and outputs')
        vbox = QVBoxLayout()
        vbox.addWidget(self.min_diameter_box)
        vbox.addWidget(self.confidence_box)
        vbox.addWidget(self.display_res_box)
        vbox.addWidget(self.reset_box)
        vbox.addWidget(self.save_box)
        self.output_widget.setLayout(vbox)


    def _setup_input_box(self):

        self.input_box = QGroupBox()
        hbox = QHBoxLayout()

        image_label = QLabel('Image: ', self)
        image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.image_layer_selection = QComboBox()
        for name in self.image_layer_names:
            self.image_layer_selection.addItem(name)
        #self.image_layer_selection.setItemText(self.image_layer_name)
        self.image_layer_selection.currentIndexChanged.connect(self._image_selection_changed)
    
        preprocess_btn = QPushButton("Preprocess")
        preprocess_btn.clicked.connect(self._on_preprocess_click)

        hbox.addWidget(image_label)
        hbox.addSpacing(5)
        hbox.addWidget(self.image_layer_selection)
        hbox.addWidget(preprocess_btn)
        self.input_box.setLayout(hbox)
        self.input_box.setStyleSheet("border: 0px")

    def _setup_model_box(self):

        self.model_box = QGroupBox()
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
        self.model_box.setLayout(hbox)
        self.model_box.setStyleSheet("border: 0px")

    def _setup_window_sizes_box(self):

        self.window_sizes_box = QGroupBox()
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
        self.window_sizes_box.setLayout(hbox)   
        self.window_sizes_box.setStyleSheet("border: 0px")  


    def _setup_downsampling_box(self):

        self.downsampling_box = QGroupBox()
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
        self.downsampling_box.setLayout(hbox)
        self.downsampling_box.setStyleSheet("border: 0px") 


    def _setup_min_diameter_box(self):

        self.min_diameter_box = QGroupBox()
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
        self.min_diameter_box.setLayout(hbox)
        self.min_diameter_box.setStyleSheet("border: 0px") 

    def _setup_confidence_box(self):

        self.confidence_box = QGroupBox()
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
        self.confidence_box.setLayout(hbox)
        self.confidence_box.setStyleSheet("border: 0px") 

    def _setup_display_res_box(self):

        self.display_res_box = QGroupBox()
        hbox = QHBoxLayout()

        self.organoid_number_label = QLabel('Number of detected organoids: 0', self)
        self.organoid_number_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        update_btn = QPushButton("Update Number")
        update_btn.clicked.connect(self._on_update_click)
    
        hbox.addWidget(self.organoid_number_label)
        hbox.addSpacing(15)
        hbox.addWidget(update_btn)
        self.display_res_box.setLayout(hbox)
        self.display_res_box.setStyleSheet("border: 0px") 

    def _setup_reset_box(self):
        self.reset_box = QGroupBox()
        hbox = QHBoxLayout()

        self.reset_btn = QPushButton("Reset Configs")
        self.reset_btn.clicked.connect(self._on_reset_click)

        self.screenshot_btn = QPushButton("Take screenshot")
        self.screenshot_btn.clicked.connect(self._on_screenshot_click)

        hbox.addWidget(self.screenshot_btn)
        hbox.addSpacing(15)
        hbox.addWidget(self.reset_btn)
        self.reset_box.setLayout(hbox)
        self.reset_box.setStyleSheet("border: 0px")

    def _setup_save_box(self):
        
        self.save_box = QGroupBox()
        hbox = QHBoxLayout()

        self.save_csv_btn = QPushButton("Save features")
        self.save_csv_btn.clicked.connect(self._on_save_csv_click)

        self.save_json_btn = QPushButton("Save boxes")
        self.save_json_btn.clicked.connect(self._on_save_json_click)

        self.save_label = QLabel('Save: ', self)
        self.save_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.output_layer_selection = QComboBox()
        '''
        # currently not supported to pre-load shapes layers
        for name in self.shape_layer_names:
            self.output_layer_selection.addItem(name)
        self.output_layer_selection.currentIndexChanged.connect(self._output_layer_selection_changed)
        '''

        hbox.addWidget(self.save_label)
        hbox.addSpacing(5)
        hbox.addWidget(self.output_layer_selection)
        hbox.addWidget(self.save_csv_btn)
        hbox.addWidget(self.save_json_btn)
        self.save_box.setLayout(hbox)
        self.save_box.setStyleSheet("border: 0px")

    def _get_layer_names(self, layer_type: layers.Layer = layers.Image) -> List[str]:
        """
        Get list of layer names of a given layer type.
        """
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if type(layer) == layer_type
        ]

        if layer_names:
            return [] + layer_names
        else:
            return []



