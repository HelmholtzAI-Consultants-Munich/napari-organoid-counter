from pathlib import Path


def init():
    global MODELS
    MODELS = {
        "faster r-cnn (DO)": {
            "filename": "faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.onnx",
            "source": "https://zenodo.org/records/18700540/files/faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.onnx",
        },
        "yolov3 (DO)": {
            "filename": "yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.onnx",
            "source": "https://zenodo.org/records/18700540/files/yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.onnx",
        },
    }

    global MODELS_DIR
    MODELS_DIR = Path.home() / ".cache/napari-organoid-counter/models"

    global MODEL_TYPE
    MODEL_TYPE = ".onnx"

    global ONNX_PREPROCESS
    ONNX_PREPROCESS = {
        "faster r-cnn (DO)": {
            "input_size": (800, 1333),
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
        },
        "yolov3 (DO)": {
            "input_size": (416, 416),
            "mean": [0.0, 0.0, 0.0],
            "std": [255.0, 255.0, 255.0],
        },
    }

    global COLOR_DEFAULT
    COLOR_DEFAULT = [1.0, 0, 1.0, 1.0]

    global COLOR_CLASS_0
    COLOR_CLASS_0 = [85 / 255, 1.0, 0, 1.0]
    global COLOR_CLASS_1
    COLOR_CLASS_1 = [0, 29 / 255, 1.0, 1.0]
    global COLOR_CLASS_2
    COLOR_CLASS_2 = [1.0, 0.65, 0, 1.0]
    global COLOR_CLASS_3
    COLOR_CLASS_3 = [128 / 256, 0, 128 / 256, 1.0]
    global COLOR_CLASS_4
    COLOR_CLASS_4 = [0.0, 1.0, 1.0, 1.0]
    global COLOR_CLASS_5
    COLOR_CLASS_5 = [1.0, 0, 0, 1.0]
    global COLOR_CLASS_6
    COLOR_CLASS_6 = [150 / 256, 75 / 256, 0 / 256, 1.0]
    global COLOR_CLASS_7
    COLOR_CLASS_7 = [0.8, 0.1, 0.6, 1.0]
    global COLOR_CLASS_8
    COLOR_CLASS_8 = [1.0, 1.0, 0.0, 1.0]
    global COLOR_CLASS_9
    COLOR_CLASS_9 = [0.3, 0.5, 1.0, 1.0]

    global COLOR_MAPPING
    COLOR_MAPPING = {
        0: (COLOR_CLASS_0, "Green"),
        1: (COLOR_CLASS_1, "Blue"),
        2: (COLOR_CLASS_2, "Orange"),
        3: (COLOR_CLASS_3, "Purple"),
        4: (COLOR_CLASS_4, "Cyan"),
        5: (COLOR_CLASS_5, "Red"),
        6: (COLOR_CLASS_6, "Brown"),
        7: (COLOR_CLASS_7, "Pink"),
        8: (COLOR_CLASS_8, "Yellow"),
        9: (COLOR_CLASS_9, "Light Blue"),
    }

    global CONFIDENCE_THRESHOLD_CLASS
    CONFIDENCE_THRESHOLD_CLASS = 0.7
