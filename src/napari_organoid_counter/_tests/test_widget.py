from napari_organoid_counter import OrganoidCounterWidget
import numpy as np

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_organoid_counter_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = OrganoidCounterWidget(viewer)

    # call our widget method
    my_widget._preprocess()
    '''TO DO'''
    '''
    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "napari has 1 layers\n"
    '''
    my_widget._on_preprocess_click()
    '''TO DO'''
    my_widget._on_run_click()
    '''TO DO'''

    my_widget._on_downsampling_changed()
    assert my_widget.downsampling == my_widget.downsampling_slider.value()

    my_widget._on_diameter_changed()
    assert my_widget.min_diameter == my_widget.min_diameter_slider.value()

    my_widget._on_sigma_changed()
    assert my_widget.sigma == my_widget.sigma_slider.value()

    my_widget._image_selection_changed()
    assert my_widget.image_layer_name == my_widget.image_layer_selection.currentText()

    my_widget._on_update_click()
    '''TO DO'''

    my_widget._on_reset_click()
    '''TO DO'''

    my_widget._on_screenshot_click()
    '''TO DO'''

    my_widget._on_save_csv_click()
    '''TO DO'''

    my_widget._on_save_json_click()
    '''TO DO'''


    
