import json
import numpy as np
from napari import layers
from pathlib import Path

from napari_organoid_counter import settings
from napari_organoid_counter._utils import get_edge_color

# Ensure settings constants are loaded when the reader is imported standalone
try:
    settings.init()
except Exception:
    pass


readable_extensions = ('.json', '.json.draft')

def get_reader(path):
    """ A basic implementation of the napari_get_reader hook specification """
    # if we know we cannot read the file, we immediately return None.
    if not any(path.endswith(ext) for ext in readable_extensions):
        return None
    # otherwise we return the *function* that can read ``path``.
    return reader_function

def reader_function_data_management(path: str) -> layers.Shapes:
    """ Reads the labels in the json file and adds a shapes layer to the napari viewer """
    # laod json
    f = open(path)
    annot = json.load(f)
    # initialise empty lists for boxes, ids and scores
    bboxes = []
    ids = []
    scores = []
    lables = []
    # for each box
    for key in annot.keys():
        # read coordinates
        x1 = round(int(float(annot[key]['x1'])))
        y1 = round(int(float(annot[key]['y1'])))
        x2 = round(int(float(annot[key]['x2'])))
        y2 = round(int(float(annot[key]['y2'])))
        # append in style readable by napari viewer 
        bboxes.append(np.array([[x1, y1],
                                [x1, y2],
                                [x2, y2],
                                [x2, y1]]))
        # and append scores and ids whihc will be used to display as text
        ids.append(int(annot[key]['box_id']))
        scores.append(float(annot[key]['confidence']))
        if 'label' in annot[key].keys():
            # if label is present, append it
            lables.append(annot[key]['label'])
        else:
            # if not, append a default label
            # this is useful for models trained on only one class
            lables.append(-1)

    # scale will adjust boxes according to physical resolution of image
    scale = (float(annot[key]['scale_x']), float(annot[key]['scale_y'])) # do only once
    # name of layer which will be created
    labels_name = 'Labels-'+Path(path).stem
    # properties used for dusplaying text
    properties = {'box_id': ids,'scores': scores, 'labels': lables}
    text_params = {'string': 'ID: {box_id}\nConf.: {scores:.2f}',
                    'size': 12,
                    'anchor': 'upper_left',}
    # edge colors for boxes
    edge_colors = get_edge_color(lables, False)
    layer_attributes = {'name': labels_name,
                        'scale': scale,
                        'properties': properties,
                        'text': text_params,
                        'face_color': 'transparent',  
                        'edge_color': edge_colors,
                        'shape_type': 'rectangle',
                        'edge_width': 12
    }
    # return data, attributes for displaying and type of layer to add to viewer
    return [(bboxes, layer_attributes, 'shapes', lables)]


def reader_function(path: str):
    """ A wrapper around the reader function to manage errors and return None if the file cannot be read """
    
    bboxes, layer_attributes, layer_type, _ = reader_function_data_management(path)
    return [(bboxes, layer_attributes, layer_type)]