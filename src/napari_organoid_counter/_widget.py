import napari
from napari.types import ImageData, LabelsData, ShapesData
#from typing import List
from magicgui import magic_factory
import numpy as np

def apply_normalization(img):
    #Normalise and return img to range 0-255
    img_min = np.min(img) # 31.3125 png 0
    img_max = np.max(img) # 2899.25 png 178
    img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    return img_norm

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
        print('counnttttttttt')


