from pathlib import Path

def init():
    
    global MODELS
    MODELS = {
        "faster r-cnn (DO)": {"filename": "faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.pth", 
                         "source": "https://zenodo.org/records/11388549/files/faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.pth"
                         },
        "ssd (DO)": {"filename": "ssd_organoid_best_coco_bbox_mAP_epoch_86.pth", 
                "source": "https://zenodo.org/records/11388549/files/ssd_organoid_best_coco_bbox_mAP_epoch_86.pth"
                },
        "yolov3 (DO)": {"filename": "yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.pth",
                   "source": "https://zenodo.org/records/11388549/files/yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.pth"
                   },
        "rtmdet (DO)":  {"filename": "rtmdet_l_organoid_best_coco_bbox_mAP_epoch_323.pth",
                    "source": "https://zenodo.org/records/11388549/files/rtmdet_l_organoid_best_coco_bbox_mAP_epoch_323.pth"
                    },
        "Binary Classification": {"filename": "best_coco_bbox_mAP_epoch_2.pth",
                        "source": "https://zenodo.org/records/14900559/files/best_coco_bbox_mAP_epoch_2.pth"
                        }
    }
    
    global MODELS_DIR
    MODELS_DIR = Path.home() / ".cache/napari-organoid-counter/models"

    global MODEL_TYPE
    MODEL_TYPE = '.pth'

    global CONFIGS
    CONFIGS = {
        "faster r-cnn (DO)": {"source": "https://zenodo.org/records/11388549/files/faster-rcnn_r50_fpn_organoid.py",
                        "destination": ".mim/configs/faster_rcnn/faster-rcnn_r50_fpn_organoid.py"
                        },
        "ssd (DO)": {"source": "https://zenodo.org/records/11388549/files/ssd_organoid.py",
                "destination": ".mim/configs/ssd/ssd_organoid.py"
                },
        "yolov3 (DO)": {"source": "https://zenodo.org/records/11388549/files/yolov3_416_organoid.py",
                "destination": ".mim/configs/yolo/yolov3_416_organoid.py"
                },
        "rtmdet (DO)":  {"source": "https://zenodo.org/records/11388549/files/rtmdet_l_organoid.py",
                    "destination": ".mim/configs/rtmdet/rtmdet_l_organoid.py"
                    },
        "Binary Classification": {"source": "https://zenodo.org/records/14900559/files/yolov3_416_organoid_two_class.py",
                        "destination": ".mim/configs/yolo/yolov3_416_organoid_two_class.py"
                        }
}
    
    # Add color definitions
    global COLOR_DEFAULT
    COLOR_DEFAULT = [1., 0, 1., 1.] # Magenta (Detection-Only)

    # Binary Classification (2 classes) 
    global COLOR_CLASS_0
    COLOR_CLASS_0 = [85 / 255, 1.0, 0, 1.0]  # Green
    
    global COLOR_CLASS_1
    COLOR_CLASS_1 = [0, 29 / 255, 1.0, 1.0]  # Blue

    # Multi-Class Palette (up to 10 classes)
    global COLOR_CLASS_2
    COLOR_CLASS_2 = [1.0, 0.65, 0, 1.0]  # Orange
    
    global COLOR_CLASS_3
    COLOR_CLASS_3 = [128/256, 0, 128/256, 1.0]  # Purple

    global COLOR_CLASS_4
    COLOR_CLASS_4 = [0.0, 1.0, 1.0, 1.0]  # Cyan

    global COLOR_CLASS_5
    COLOR_CLASS_5 = [1.0, 0, 0, 1.0]  # Red

    global COLOR_CLASS_6
    COLOR_CLASS_6 = [150/256, 75/256, 0/256, 1.0]  # Brown

    global COLOR_CLASS_7
    COLOR_CLASS_7 = [0.8, 0.1, 0.6, 1.0]  # Pink

    global COLOR_CLASS_8
    COLOR_CLASS_8 = [1.0, 1.0, 0.0, 1.0]  # Yellow

    global COLOR_CLASS_9
    COLOR_CLASS_9 = [0.3, 0.5, 1.0, 1.0]  # Light Blue
    



