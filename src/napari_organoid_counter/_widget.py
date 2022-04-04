import napari
from napari.types import ImageData, LabelsData, ShapesData
#from typing import List
from magicgui import magic_factory
import numpy as np

import math
import numpy as np
from aicsimageio import AICSImage
from scipy import ndimage as ndi
from skimage.measure import block_reduce
from skimage.feature import canny
from skimage.measure import regionprops,label
from skimage.morphology import remove_small_objects, erosion, disk
import cv2

def apply_normalization(img):
    #Normalise and return img to range 0-255
    img_min = np.min(img) # 31.3125 png 0
    img_max = np.max(img) # 2899.25 png 178
    img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    return img_norm


def count_organoids(img_original, img_scale, downsampling, min_diameter_um, sigma, low_threshold, high_threshold, background_intensity=40):
    img = block_reduce(img_original, block_size=(downsampling, downsampling), func=np.mean)   
    img_scale = [downsampling*x for x in img_scale] # update resolutions
    # normalise
    img = apply_normalization(img) # DONE IN PRE_PROCESS
    # get mask of well and background
    mask = np.where(img<background_intensity,False,True)
    # find edges in image
    edges = canny(
                image=img,
                sigma=sigma,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                mask=mask)
    # dilate edges
    edges = ndi.binary_dilation(edges)
    # fill holes
    filled = ndi.binary_fill_holes(edges)
    filled = erosion(filled, disk(2))
    labels = label(filled)
    region = regionprops(labels)
    # remove objects larger than 30% of the image
    for prop in region:
        if prop.area > 0.3*img.shape[0]*img.shape[1]:
            filled[prop.coords] = 0
    # get min organoid size in pixels
    min_radius_um = min_diameter_um//2
    min_area = math.pi * min_radius_um**2 
    min_area_pix = min_area / (img_scale[0] * img_scale[1])
    min_size_pix = round(min_area_pix)
    filled = remove_small_objects(filled, min_size_pix)
    segmentation = label(filled)
    return segmentation

@magic_factory(preprocess_button=dict(widget_type="PushButton", text="Preprocess"),
            run_button=dict(widget_type="PushButton", text="Run Organoid Counter"),
            downsampling={"widget_type": "Slider", "min": 1, "max": 10},
            min_diameter={"widget_type": "Slider", "min": 10, "max": 100},
            sigma={"widget_type": "Slider", "min": 1, "max": 10},
            auto_call=True)
# this function is called each time a parameter is changed when auto_call=True
def organoid_counter_widget(image_layer: napari.layers.Image,
                        preprocess_button,
                        run_button,
                        downsampling: int=4,
                        min_diameter: int=30,
                        sigma: int=2) -> None: #-> List[napari.types.LayerDataTuple]:
    print('A', image_layer.scale)
    @organoid_counter_widget.preprocess_button.clicked.connect
    def preprocess(event=None):
        img = np.squeeze(image_layer.data)
        img = img.astype(np.float64)
        img = apply_normalization(img)
        image_layer.data = img
        print('HE')
        
    @organoid_counter_widget.run_button.clicked.connect
    def run_orgacount(event=None):
        segmentation = count_organoids(image_layer.data, image_layer.scale, downsampling, min_diameter, sigma)


@magic_factory(update_button=dict(widget_type="PushButton", text="Update Results"),
            save_button=dict(widget_type="PushButton", text="Save Results"),
            screenshot_button=dict(widget_type="PushButton", text="Take screenshot"),
            auto_call=True)
def update_and_save_output(update_button, save_button):
    @update_and_save_output.update_button.clicked.connect
    def update_orga_number(event=None):
        pass
    @update_and_save_output.save_button.clicked.connect
    def save_orga_results(event=None):
        pass
    @update_and_save_output.screenshot_button.clicked.connect
    def take_screenshot(event=None):
        pass