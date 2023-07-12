from pathlib import Path

def init():
    
    global MODELS
    MODELS = {
        "default": {"filename": "model_v1.ckpt", "source": "https://zenodo.org/record/7708763/files/model_v1.ckpt"}
    }
    
    global MODELS_DIR
    MODELS_DIR = Path.home() / ".cache/napari-organoid-counter/models"

    global MODEL_TYPE
    MODEL_TYPE = '.ckpt'

