import os
import numpy as np
from skimage.io import imsave
import csv
import json
from datetime import datetime

from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QPushButton, QVBoxLayout, QWidget, QSlider, QLabel)
from napari.utils.notifications import show_info
from ._orgacount import *

class OrganoidCounterWidget(QWidget):
    # the widget of the organoid counter - documentation to be added
    def __init__(self, 
                napari_viewer,
                downsampling=4,
                min_diameter=30,
                sigma=2):
        super().__init__()
        self.viewer = napari_viewer
        # this has to be changed if we later add more images it needs to be updated 
        layer_names = [layer.name for layer in self.viewer.layers]
        self.original_images = {}
        self.original_contrast = {}
        if layer_names: 
            self.image_layer_name = layer_names[0]
            for layer_name in layer_names:
                self.original_images[layer_name] = self.viewer.layers[layer_name].data
                self.original_contrast[layer_name] = self.viewer.layers[self.image_layer_name].contrast_limits
        else: self.image_layer_name = None
        self.bboxes = []

        self.image_layer_selection = QComboBox()
        for name in layer_names:
            self.image_layer_selection.addItem(name)
        #self.image_layer_selection.setItemText(self.image_layer_name)
        self.image_layer_selection.currentIndexChanged.connect(self._image_selection_changed)

        preprocess_btn = QPushButton("Preprocess")
        preprocess_btn.clicked.connect(self._on_preprocess_click)

        run_btn = QPushButton("Run Organoid Counter")
        run_btn.clicked.connect(self._on_run_click)

        self.downsampling = downsampling
        self.downsampling_slider = QSlider(Qt.Horizontal)
        self.downsampling_slider.setMinimum(1)
        self.downsampling_slider.setMaximum(10)
        self.downsampling_slider.setSingleStep(1)
        self.downsampling_slider.setValue(self.downsampling)
        self.downsampling_slider.valueChanged.connect(self._on_downsampling_changed)

        self.downsampling_label = QLabel('Downsampling: 4', self)
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

        self.sigma = sigma
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(10)
        self.sigma_slider.setSingleStep(1)
        self.sigma_slider.setValue(2)
        self.sigma_slider.valueChanged.connect(self._on_sigma_changed)

        self.sigma_label = QLabel('Sigma: 2', self)
        self.sigma_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        #self.min_diameter_label.setMinimumWidth(80)

        self.sigma_box = QWidget()
        self.sigma_box.setLayout(QHBoxLayout())
        self.sigma_box.layout().addWidget(self.sigma_label)
        self.sigma_box.layout().addSpacing(15)
        self.sigma_box.layout().addWidget(self.sigma_slider)

        self.update_btn = QPushButton("Update Results")
        self.update_btn.clicked.connect(self._on_update_click)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._on_reset_click)

        self.screenshot_btn = QPushButton("Take screenshot")
        self.screenshot_btn.clicked.connect(self._on_screenshot_click)

        self.save_csv_btn = QPushButton("Save features")
        self.save_csv_btn.clicked.connect(self._on_save_csv_click)

        self.save_json_btn = QPushButton("Save boxes")
        self.save_json_btn.clicked.connect(self._on_save_json_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.image_layer_selection)
        self.layout().addWidget(preprocess_btn)
        self.layout().addWidget(run_btn)
        self.layout().addWidget(self.downsampling_box)
        self.layout().addWidget(self.min_diameter_box)
        self.layout().addWidget(self.sigma_box)
        self.layout().addWidget(self.update_btn)
        self.layout().addWidget(self.reset_btn)
        self.layout().addWidget(self.screenshot_btn)
        self.layout().addWidget(self.save_csv_btn)
        self.layout().addWidget(self.save_json_btn)

    def _preprocess(self):
        img = np.squeeze(self.original_images[self.image_layer_name]) #self.viewer.layers[self.image_layer_name].data)
        img = img.astype(np.float64)
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
            img_scale = self.viewer.layers[self.image_layer_name].scale
            segmentation = count_organoids(img_data, 
                                            img_scale, 
                                            self.downsampling, 
                                            self.min_diameter,
                                            self.sigma)
            _, _, self.bboxes = setup_bboxes(segmentation)
            img_data = add_text_to_img(img_data, len(self.bboxes))
            self.viewer.layers[self.image_layer_name].data = img_data
            layer_names = [layer.name for layer in self.viewer.layers]
            seg_layer_name = 'Organoids '+self.image_layer_name
            if seg_layer_name in layer_names:
                self.viewer.layers[seg_layer_name].data = self.bboxes
            else:
                self.viewer.add_shapes(self.bboxes, 
                                        name=seg_layer_name,
                                        scale=self.viewer.layers[self.image_layer_name].scale,
                                        face_color='transparent',  
                                        edge_color='magenta',
                                        shape_type='rectangle',
                                        edge_width=12) # warning generated here

    def _on_downsampling_changed(self):
        self.downsampling = self.downsampling_slider.value()
        self.downsampling_label.setText('Downsampling: '+str(self.downsampling))

    def _on_diameter_changed(self):
        self.min_diameter = self.min_diameter_slider.value()
        self.min_diameter_label.setText('Min. Diameter: '+str(self.min_diameter))

    def _on_sigma_changed(self):
        self.sigma = self.sigma_slider.value()
        self.sigma_label.setText('Sigma: '+str(self.sigma))

    def _image_selection_changed(self):
        self.image_layer_name = self.image_layer_selection.currentText()

    def _on_update_click(self):
        if not self.image_layer_name: show_info('Please load an image first and try again!')
        else:
            self._preprocess()
            img_data = self.viewer.layers[self.image_layer_name].data # get pre-processed image!!!
            bboxes = self.viewer.layers['Organoids '+self.image_layer_name].data
            img_data = add_text_to_img(img_data, len(bboxes))
            self.viewer.layers[self.image_layer_name].data = img_data

    def _on_reset_click(self):
        # reset params
        self.downsampling = 4
        self.downsampling_slider.setValue(self.downsampling)
        self.min_diameter=30
        self.min_diameter_slider.setValue(self.min_diameter)
        self.sigma=2
        self.sigma_slider.setValue(self.sigma)
        if self.image_layer_name:
            # reset to original image and **TO-DO** remove layer of results
            self.viewer.layers[self.image_layer_name].data = self.original_images[self.image_layer_name]
            self.viewer.layers[self.image_layer_name].contrast_limits = self.original_contrast[self.image_layer_name]

    def _on_screenshot_click(self):
        screenshot=self.viewer.screenshot()
        if not self.image_layer_name: screenshot_name = datetime.now().strftime("%d%m%Y%H%M%S")+'screenshot.png'
        else: screenshot_name = self.image_layer_name+datetime.now().strftime("%d%m%Y%H%M%S")+'_screenshot.png'
        imsave(screenshot_name, screenshot)

    def _on_save_csv_click(self): 
        if not self.bboxes: show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
        else:
            data_csv = []
            if not os.path.isdir('annotations'): os.mkdir = 'annotations'
            bboxes = self.viewer.layers['Organoids '+self.image_layer_name].data
            # save diameters and area of organoids (approximated as ellipses)
            for i, bbox in enumerate(bboxes):
                d1 = abs(bbox[0][0] - bbox[2][0]) * self.viewer.layers[self.image_layer_name].scale[0]
                d2 = abs(bbox[0][1] - bbox[2][1]) * self.viewer.layers[self.image_layer_name].scale[0]
                area = math.pi * d1 * d2
                data_csv.append([i, round(d1,3), round(d2,3), round(area,3)])
            #write diameters and area to csv
            output_path_csv = os.path.join('annotations', self.image_layer_name+'.csv')
            with open(output_path_csv, 'w') as f:
                write = csv.writer(f, delimiter=';')
                write.writerow(['OrganoidID', 'D1[um]','D2[um]', 'Area [um^2]'])
                write.writerows(data_csv)
        
    def _on_save_json_click(self):
        if not self.bboxes: show_info('No organoids detected! Please run auto organoid counter or run algorithm first and try again!')
        else:
            data_json = {} 
            if not os.path.isdir('annotations'): os.mkdir = 'annotations'
            output_path_json = os.path.join('annotations', self.image_layer_name+'.json')
            bboxes = self.viewer.layers['Organoids '+self.image_layer_name].data
            for i, bbox in enumerate(bboxes):
                data_json.update({str(i): [list(bboxit) for bboxit in bbox]})
            #write bbox coordinates to json
            with open(output_path_json, 'w') as outfile:
                json.dump(data_json, outfile)  



