import json
import numpy as np
from napari_organoid_counter import reader_function, get_reader
import pytest

# Define a fixture to create a temporary file with JSON data
@pytest.fixture
def json_file(tmp_path):
    # tmp_path is a pytest fixture

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

    # write some fake data using your supported file format
    json_path = tmp_path / 'reader_test.json'
    with open(json_path, 'w') as outfile:
        json.dump(original_data, outfile)

    return json_path


def test_reader(json_file):
    """Testing the reader part of the plugin"""

    # Call the reader function with the path to the temporary JSON file
    result = reader_function(str(json_file))
    # Verify the result
    assert len(result) == 1  # Check that a single layer is returned

    # Unpack the returned tuple
    bboxes, layer_attributes, layer_type = result[0]
    
    # Verify the bounding boxes
    assert isinstance(bboxes, list) and len(bboxes) == 2
    for bbox in bboxes:
        assert isinstance(bbox, np.ndarray) and bbox.shape == (4, 2)

    # Verify the layer attributes
    assert isinstance(layer_attributes, dict)
    assert layer_attributes['name'] == 'Labels-reader_test'
    assert isinstance(layer_attributes['scale'], tuple)
    assert len(layer_attributes['scale']) == 2
    assert isinstance(layer_attributes['properties'], dict)
    assert len(layer_attributes['properties']['box_id']) == 2
    assert len(layer_attributes['properties']['scores']) == 2

    # Verify the layer type
    assert isinstance(layer_type, str) and layer_type == 'shapes'

def test_get_reader(json_file):
    # Call the get_reader function with the temporary JSON file path
    reader = get_reader(str(json_file))
    # Verify that the returned value is a callable
    assert callable(reader)

def test_get_reader_pass():
    reader = get_reader('fake.file')
    assert reader is None # Verify that None is returned for unsupported file extensions