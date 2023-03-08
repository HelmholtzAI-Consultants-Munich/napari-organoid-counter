import json
import numpy as np
from napari import layers
from pathlib import Path

readable_extensions = '.json'

def get_reader(path):
    """A basic implementation of the napari_get_reader hook specification."""
    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(readable_extensions):
        return None
    # otherwise we return the *function* that can read ``path``.
    return reader_function

def reader_function(path: str) -> layers.Shapes:
    f = open(path)
    annot = json.load(f)
    bboxes = []

    for key in annot.keys():
        x1 = round(annot[key]['x1'])
        y1 = round(annot[key]['y1'])
        x2 = round(annot[key]['x2'])
        y2 = round(annot[key]['y2'])
        scale = (annot[key]['scale_x'], annot[key]['scale_y'])
        bboxes.append(np.array([[x1, y1],
                                [x1, y2],
                                [x2, y2],
                                [x2, y1]]))
    labels_name = 'Labels-'+Path(path).stem
    layer_attributes = {'name': labels_name,
                        'scale': scale,
                        'face_color': 'transparent',  
                        'edge_color': 'magenta',
                        'shape_type': 'rectangle',
                        'edge_width': 12
    }

    return [(bboxes, layer_attributes, 'shapes')]