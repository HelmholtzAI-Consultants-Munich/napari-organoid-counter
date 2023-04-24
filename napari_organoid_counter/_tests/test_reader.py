import json
import numpy as np
from napari_organoid_counter import get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """Testing the reader part of the plugin"""

    # write some fake data using your supported file format
    my_test_file = str(tmp_path / 'myfile.json')
    bboxes = [np.array([[2000,2000], [2000,2500], [2500,2500], [2500,2000]]),
              np.array([[3000,3000], [3000,3500], [3500,3500], [3500,3000]])]

    original_data = {
        '1':{
            'box_id': '1',
            'x1': 2000,
            'x2': 2500,
            'y1': 2000,
            'y2': 2500,
            'confidence': 0.8,
            'scale_x': 0.9,
            'scale_y': 0.9,
        },
        '2':{
            'box_id': '2',
            'x1': 3000,
            'x2': 3500,
            'y1': 3000,
            'y2': 3500,
            'confidence': 0.85,
            'scale_x': 0.9,
            'scale_y': 0.9,
        }
    }
    with open(my_test_file, 'w') as outfile:
        outfile.write(json.dumps(original_data))

    # try to read it back in
    reader = get_reader(my_test_file) #reader should return [(bboxes, layer_attributes, 'shapes')]
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0 
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple)==3

    # make sure it's the same as it started
    data = layer_data_tuple[0]
    for idx, bbox in enumerate(bboxes):
        assert (data[idx]==bbox).all()

    # and that correct attributes and layer type is set
    attributes = layer_data_tuple[1]
    assert attributes['name']=='Labels-myfile'
    assert attributes['scale']==(original_data['2']['scale_x'], original_data['2']['scale_y'])
    assert attributes['properties']['scores'][-1]==original_data['2']['confidence']
    
    layer_type = layer_data_tuple[2]
    assert isinstance(layer_type, str) and layer_type=='shapes'


def test_get_reader_pass():
    reader = get_reader('fake.file')
    assert reader is None