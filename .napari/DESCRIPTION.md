## Description

A napari plugin to automatically count lung organoids from microscopy imaging data. A Faster R-CNN model was trained on patches of microscopy data. Model inference is run using a sliding window approach, with a 50% overlap and the option for predicting on multiple window sizes and scales, the results of which are then merged using NMS.

![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/main/readme-content/demo-plugin-v2.gif)

## What's new in v2?
Here is a list of the main changes v2 of napari-organoid-counter offers:
* Use of Faster R-CNN model for object detection 
* Pyramid model inference with a sliding window approach and tunable parameters for window size and window downsampling rate
* Model confidence added as tunable parameter
* Allow to load and correct existing annotations (note: these must have been saved previously from v2 of this plugin)
* Object ID along with model confidence displayed in the viewer - this can now be related to box id in csv file of extracted features
* _Fixed:_ box thickness changing at different donwsampling rates
* Possibility to work interactively with different shape layers at the same time, go back adjust parameters and switch between shape layers from layer list selection

Technical Extensions:
* Allows for Python 3.10
* Extensive testing

## Installation

You can install `napari-organoid-counter` via [pip](https://pypi.org/project/napari-organoid-counter/):

    pip install napari-organoid-counter


To install latest development version :

    pip install git+https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter.git
 

## Quickstart

The use of the napari-organoid-counter plugin is straightforward. Here is a step-to-step guide of the standard workflow:
1. You can load the image or images you wish to process into the napari viewer with a simple drag and drop
2. You can then select the layer you wish to work on by the drop-down box at the top of the input configurations
3. To improve the way the image is visualised you can pre-process them by clicking the _Preprocess_ button and the image layer will automatically be updated with the result
4. If you have a Faster R-CNN model you wish to use for the prediction, you can browser and select this by clicking on the _Choose_ button. Otherwise, the default model will be automatically downloaded from [here](https://zenodo.org/record/7708763#.ZDe6pS8Rpqs). Note that your own model must follow the specifications described here _(TODO)_.
5. You can adjust the _Window sizes_ and _Downsampling_ parameters to define the window size in the sliding window inference and the downsampling that is performed on the image. If you have multiple objects with different sizes, it might be good to set multiple window sizes, with corresponding downsampling rates. You can seperate these with a comma in the text box (e.g. ```2048, 512```). After you have set _Window sizes_ and _Downsampling_ hit **Enter** for each for the changes to be accepted. 

**_Downsampling parameter:_** To detect large organoids (and ignore smaller structures) you may need to increase the downsampling rate, whereas if your organoids are small and are being missed by the algorithm, consider reducing the downsampling rate. 

**_Window size parameter:_** The window size can also impact the number of objects detected: typically a ratio of 512 to 1 between window size and downsampling rate would give optimal results, while larger window sizes would lead to a drop in performance. However, please note that small window sizes will signicantly impact the runtime of the algorithm.

6. By clicking the _Run Organoid Counter_ button the detection algorithm will run and a new shapes layer will be added to the viewer, with bounding boxes are placed around the detected organoid. You can add, edit or remove boxes using the _layer controls_ window (top left). The _Number of detected organoids_ will show you the number of organoids in the layer in real time. You can switch between viewing the box ids and model confidence for each box by toggling the _display text_ box in the _layer controls_ window. Boxes added by the user will by default have a confidence of 1.
7. If you feel that your model is over- or under-predicting you can use the _Model confidence_ scroll bar and select the value which best suits your problem. Default confidence is set to 0.8.
8. If you objects are typically bigger or smaller than those displayed you can use the _Minimum Diameter_ slider to set the minimum diameter of your objects. Default value is 30 um.
9. After you are happy with the detection results (and your manual corrections if any), you can select which shapes layer results to save via the dropdown box. To save the bounding box coordinates (along with the box id, model confidence for that box, and x and y scale for that image) as a json file click _Save boxes_ and select where to save your file. The features can be saved in a csv file by clicking on the _Save features_. The saved features will include the organoid id, the two lengths of the boundig box (which corresponds to the min. and max. diameter of the organoid, approximated as an ellipse), the area of the organoid (again approximated as an ellipse).
10. Additional options: The _Take Screenshot_ button has the same functionality as _File -> Save screenshot_. The _Reset Configs_ button will reset the image and all parameters to the original settings. To save features of the detected organoids (diameters when approximating organoid as an ellipse and organoid area) in a csv file click _Save features_. 


## Getting Help

If you encounter any problems, please [file an issue](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/issues) along with a detailed description.

## Intended Audience & Supported Data

This plugin has been developed and tested with 2D CZI microscopy images of lunch organoids. The images have been previously converted from a 3D stack to 2D using an extended focus algorithm. This plugin only supports single channel grayscale images. This plugin may be used as a baseline for developers who wish to extend the plugin to work with other types of input images and/or improve the detection algorithm. 

## Dependencies

```napari-organoid-counter``` uses the ```napari-aicsimageio```<sup>[1]</sup> <sup>[2]</sup> plugin for reading and processing CZI images.

[1] Eva Maxfield Brown, Dan Toloudis, Jamie Sherman, Madison Swain-Bowden, Talley Lambert, AICSImageIO Contributors (2021). AICSImageIO: Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python [Computer software]. GitHub. https://github.com/AllenCellModeling/aicsimageio

[2] Eva Maxfield Brown, Talley Lambert, Peter Sobolewski, Napari-AICSImageIO Contributors (2021). Napari-AICSImageIO: Image Reading in Napari using AICSImageIO [Computer software]. GitHub. https://github.com/AllenCellModeling/napari-aicsimageio

## How to Cite
If you use this plugin for your work, please cite it using the following:

> Christina Bukas, Harshavardhan Subramanian, & Marie Piraud. (2023). HelmholtzAI-Consultants-Munich/napari-organoid-counter: v0.2.0 (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.7859571
> 
bibtex:
```
@software{christina_bukas_2022_6457904,
  author       = {Christina Bukas, Harshavardhan Subramanian, & Marie Piraud},
  title        = {{HelmholtzAI-Consultants-Munich/napari-organoid- 
                   counter: second release of the napari plugin for lung
                   organoid counting}},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.2.0},
  doi          = {10.5281/zenodo.7859571},
  url          = {https://doi.org/10.5281/zenodo.7859571}
}
```


