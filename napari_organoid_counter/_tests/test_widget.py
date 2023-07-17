from napari_organoid_counter import OrganoidCounterWidget
import numpy as np
from skimage import draw
from napari import layers

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_organoid_counter_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    test_img = np.zeros((1000,1000))
    x1, y1 = draw.disk((500,500), 400)
    x2, y2 = draw.disk((400,400), 40)
    x3, y3 = draw.disk((600,600), 10)
    x4, y4 = draw.disk((650,400), 100)
    test_img[x1,y1] = 3000
    test_img[x2,y2] = 1500
    test_img[x3,y3] = 2000
    test_img[x4,y4] = 1000
    viewer.add_image(test_img, name='Test')

    # create our widget, passing in the viewer
    my_widget = OrganoidCounterWidget(viewer, 
                                      window_sizes=[500],
                                      downsampling=[2],
                                      window_overlap=1)

    # call preprocessing - remove duplicate here?
    my_widget._preprocess()
    img = my_widget.viewer.layers['Test'].data
    assert np.max(img) <= 255.
    assert np.min(img) >= 0.

    my_widget._on_preprocess_click()
    img = my_widget.viewer.layers['Test'].data
    assert np.max(img) <= 255.
    assert np.min(img) >= 0.

    # test that organoid counting algorithm has run and new layer with res has been added to viewer
    my_widget.organoiDL.download_model(my_widget.model_name)
    my_widget._on_run_click()
    layer_names = [layer.name for layer in  my_widget.viewer.layers]
    assert 'Labels-Test' in layer_names
    
    # test that class attributes are updated when user changes values from GUI
    my_widget._on_diameter_slider_changed()
    assert my_widget.min_diameter == my_widget.min_diameter_slider.value()

    my_widget._on_image_selection_changed()
    assert my_widget.image_layer_name == my_widget.image_layer_selection.currentText()

    # test that number of organoids is updated after manual corrections

    # test that reset button resets all parameters to default settings 
    my_widget._on_reset_click()
    assert my_widget.min_diameter==30
    assert my_widget.min_diameter_slider.value()==30
    assert my_widget.confidence==0.8
    assert my_widget.confidence_slider.value()==80

    #my_widget._on_screenshot_click()
    #'''TO DO'''

    #my_widget._on_save_csv_click()
    #'''TO DO'''

    #my_widget._on_save_json_click()
    #'''TO DO'''

    my_widget._get_layer_names()
    layer_names = [layer.name for layer in my_widget.viewer.layers if type(layer)==layers.Image]
    assert len(layer_names) == 1
    assert layer_names[0] == 'Test'
    my_widget._get_layer_names(layer_type=layers.Shapes)
    layer_names = [layer.name for layer in my_widget.viewer.layers if type(layer)==layers.Shapes]
    assert len(layer_names) == 1
    assert layer_names[0] == 'Labels-Test'