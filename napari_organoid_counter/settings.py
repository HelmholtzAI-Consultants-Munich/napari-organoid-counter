from pathlib import Path

def init():
    
    global MODELS
    MODELS = {
        "faster r-cnn": {"filename": "faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.pth", 
                         "source": "https://zenodo.org/records/11388549/files/faster-rcnn_r50_fpn_organoid_best_coco_bbox_mAP_epoch_68.pth"
                         },
        "ssd": {"filename": "ssd_organoid_best_coco_bbox_mAP_epoch_86.pth", 
                "source": "https://zenodo.org/records/11388549/files/ssd_organoid_best_coco_bbox_mAP_epoch_86.pth"
                },
        "yolov3": {"filename": "yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.pth",
                   "source": "https://zenodo.org/records/11388549/files/yolov3_416_organoid_best_coco_bbox_mAP_epoch_27.pth"
                   },
        "rtmdet":  {"filename": "rtmdet_l_organoid_best_coco_bbox_mAP_epoch_323.pth",
                    "source": "https://zenodo.org/records/11388549/files/rtmdet_l_organoid_best_coco_bbox_mAP_epoch_323.pth"
                    },
    }
    
    global MODELS_DIR
    MODELS_DIR = Path.home() / ".cache/napari-organoid-counter/models"

    global MODEL_TYPE
    MODEL_TYPE = '.pth'

    global CONFIGS
    CONFIGS = {
        "faster r-cnn": {"source": "https://zenodo.org/records/11388549/files/faster-rcnn_r50_fpn_organoid.py",
                        "destination": "./mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_organoid.py"
                        },
        "ssd": {"source": "https://zenodo.org/records/11388549/files/ssd_organoid.py",
                "destination": "./configs/ssd/ssd_organoid.py"
                },
        "yolov3": {"source": "https://zenodo.org/records/11388549/files/yolov3_416_organoid.py",
                "destination": "./mmdetection/configs/yolo/yolov3_416_organoid.py"
                },
        "rtmdet":  {"source": "https://zenodo.org/records/11388549/files/rtmdet_l_organoid.py",
                    "destination": "./mmdetection/configs/rtmdet/rtmdet_l_organoid.py"
                    }
}



