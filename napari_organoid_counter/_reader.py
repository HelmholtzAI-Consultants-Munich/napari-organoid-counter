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
    '''
    Reads the labels in the json file and adds a shapes layer to the napari viewer
    '''
    f = open(path)
    annot = json.load(f)
    bboxes = []
    ids = []
    scores = []
    for key in annot.keys():
        x1 = round(int(float(annot[key]['x1'])))
        y1 = round(int(float(annot[key]['y1'])))
        x2 = round(int(float(annot[key]['x2'])))
        y2 = round(int(float(annot[key]['y2'])))
        bboxes.append(np.array([[x1, y1],
                                [x1, y2],
                                [x2, y2],
                                [x2, y1]]))
        
        ids.append(int(annot[key]['box_id']))
        scores.append(float(annot[key]['confidence']))
    
    scale = (float(annot[key]['scale_x']), float(annot[key]['scale_y'])) # do only once
    labels_name = 'Labels-'+Path(path).stem
    properties = {'box_id': ids,'scores': scores}
    text_params = {'string': 'ID: {box_id}\nConf.: {scores:.2f}',
                    'size': 12,
                    'anchor': 'upper_left',}
    layer_attributes = {'name': labels_name,
                        'scale': scale,
                        'properties': properties,
                        'text': text_params,
                        'face_color': 'transparent',  
                        'edge_color': 'magenta',
                        'shape_type': 'rectangle',
                        'edge_width': 12
    }

    return [(bboxes, layer_attributes, 'shapes')]