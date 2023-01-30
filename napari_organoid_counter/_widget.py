import csv
import json
from skimage.io import imsave
from datetime import datetime
from typing import List

from napari import layers
from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QPushButton, QVBoxLayout, QWidget, QSlider, QLabel, QFileDialog)
from napari.utils.notifications import show_info
from ._orgacount import OrganoiDL, apply_normalization

import warnings
warnings.filterwarnings("ignore")

class OrganoidCounterWidget(QWidget):
    # the widget of the organoid counter - documentation to be added
    def __init__(self, 
                napari_viewer,
                downsampling=2,
                min_diameter=30,
                confidence=0.05):
        super().__init__()
        self.viewer = napari_viewer
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

        self.image_label = QLabel('Image: ', self)
        self.image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.image_layer_selection = QComboBox()
        for name in self.image_layer_names:
            self.image_layer_selection.addItem(name)
        #self.image_layer_selection.setItemText(self.image_layer_name)
        self.image_layer_selection.currentIndexChanged.connect(self._image_selection_changed)
        self.cur_shapes = None
        
        preprocess_btn = QPushButton("Preprocess")
        preprocess_btn.clicked.connect(self._on_preprocess_click)

        self.input_box = QWidget()
        self.input_box.setLayout(QHBoxLayout())
        self.input_box.layout().addWidget(self.image_label)
        self.input_box.layout().addSpacing(5)
        self.input_box.layout().addWidget(self.image_layer_selection)
        self.input_box.layout().addWidget(preprocess_btn)

        self.downsampling = downsampling
        self.downsampling_slider = QSlider(Qt.Horizontal)
        self.downsampling_slider.setMinimum(1)
        self.downsampling_slider.setMaximum(10)
        self.downsampling_slider.setSingleStep(1)
        self.downsampling_slider.setValue(self.downsampling)
        self.downsampling_slider.valueChanged.connect(self._on_downsampling_changed)

        self.downsampling_label = QLabel('Downsampling: 2', self)
        self.downsampling_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        #self.downsampling_label.setMinimumWidth(80)

        self.downsampling_box = QWidget()
        self.downsampling_box.setLayout(QHBoxLayout())
        self.downsampling_box.layout().addWidget(self.downsampling_label)
        self.downsampling_box.layout().addSpacing(15)
        self.downsampling_box.layout().addWidget(self.downsampling_slider)

        self.min_diameter = min_diameter
        self.min_diameter_slider = QSlider(Qt.Horizontal)
        self.min_diameter_slider.setMinimum(10)
        self.min_diameter_slider.setMaximum(100)
        self.min_diameter_slider.setSingleStep(10)
        self.min_diameter_slider.setValue(30)
        self.min_diameter_slider.valueChanged.connect(self._on_diameter_changed)

        self.min_diameter_label = QLabel('Min. Diameter: 30', self)
        self.min_diameter_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        #self.min_diameter_label.setMinimumWidth(80)

        self.min_diameter_box = QWidget()
        self.min_diameter_box.setLayout(QHBoxLayout())
        self.min_diameter_box.layout().addWidget(self.min_diameter_label)
        self.min_diameter_box.layout().addSpacing(15)
        self.min_diameter_box.layout().addWidget(self.min_diameter_slider)

        self.confidence = confidence
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(1)
        self.confidence_slider.setSingleStep(0.05)
        self.confidence_slider.setValue(0.05)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)

        self.confidence_label = QLabel('Confidence: 0.05', self)
        self.confidence_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        #self.min_diameter_label.setMinimumWidth(80)

        self.confidence_box = QWidget()
        self.confidence_box.setLayout(QHBoxLayout())
        self.confidence_box.layout().addWidget(self.confidence_label)
        self.confidence_box.layout().addSpacing(15)
        self.confidence_box.layout().addWidget(self.confidence_slider)

        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)

        self.organoid_number_label = QLabel('Number of detected organoids: 0', self)
        self.organoid_number_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.update_btn = QPushButton("Update Number")
        self.update_btn.clicked.connect(self._on_update_click)
        
        self.display_res_box = QWidget()
        self.display_res_box.setLayout(QHBoxLayout())
        self.display_res_box.layout().addWidget(self.organoid_number_label)
        self.display_res_box.layout().addSpacing(15)
        self.display_res_box.layout().addWidget(self.update_btn)

        self.reset_btn = QPushButton("Reset Configs")
        self.reset_btn.clicked.connect(self._on_reset_click)

        self.screenshot_btn = QPushButton("Take screenshot")
        self.screenshot_btn.clicked.connect(self._on_screenshot_click)

        self.reset_box = QWidget()
        self.reset_box.setLayout(QHBoxLayout())
        self.reset_box.layout().addWidget(self.screenshot_btn)
        self.reset_box.layout().addSpacing(15)
        self.reset_box.layout().addWidget(self.reset_btn)

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
    
        self.save_box = QWidget()
        self.save_box.setLayout(QHBoxLayout())
        self.save_box.layout().addWidget(self.save_label)
        self.save_box.layout().addSpacing(5)
        self.save_box.layout().addWidget(self.output_layer_selection)
        self.save_box.layout().addWidget(self.save_csv_btn)
        self.save_box.layout().addWidget(self.save_json_btn)
        
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.input_box)
        self.layout().addWidget(self.downsampling_box)
        self.layout().addWidget(self.min_diameter_box)
        self.layout().addWidget(self.confidence_box)
        self.layout().addWidget(run_btn)
        self.layout().addWidget(self.display_res_box)
        self.layout().addWidget(self.reset_box)
        self.layout().addWidget(self.save_box)

        self.organoiDL = OrganoiDL()
        
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

    def _on_preprocess_click(self):
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else: self._preprocess()


    def _on_run_click(self):
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else:
            self._preprocess()
            img_data = self.viewer.layers[self.image_layer_name].data
            #img_scale = self.viewer.layers[self.image_layer_name].scale
  
            bboxes = self.organoiDL.run(img_data, 
                                        self.downsampling,
                                        self.min_diameter,
                                        self.confidence)
            
            new_text = 'Number of detected organoids: '+str(len(bboxes))
            self.organoid_number_label.setText(new_text)
            seg_layer_name = 'Organoids '+self.image_layer_name
            if seg_layer_name in self.shape_layer_names:
                self.viewer.layers[seg_layer_name].data = bboxes
            else:
                self.viewer.add_shapes(bboxes, 
                                        name=seg_layer_name,
                                        scale=self.viewer.layers[self.image_layer_name].scale,
                                        face_color='transparent',  
                                        edge_color='magenta',
                                        shape_type='rectangle',
                                        edge_width=12) # warning generated here
            self.cur_shapes = seg_layer_name

    def _on_downsampling_changed(self):
        self.downsampling = self.downsampling_slider.value()
        self.downsampling_label.setText('Downsampling: '+str(self.downsampling))

    def _on_diameter_changed(self):
        self.min_diameter = self.min_diameter_slider.value()
        self.min_diameter_label.setText('Min. Diameter: '+str(self.min_diameter))

    def _on_confidence_changed(self):
        self.confidence = self.confidence_slider.value()
        self.confidence_label.setText('Confidence: '+str(self.confidence))

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
        self.downsampling = 2
        self.downsampling_slider.setValue(self.downsampling)
        self.min_diameter=30
        self.min_diameter_slider.setValue(self.min_diameter)
        self.confidence=0.05
        self.confidence_slider.setValue(self.confidence)
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



