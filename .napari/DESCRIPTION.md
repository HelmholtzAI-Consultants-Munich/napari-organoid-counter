## Description

A napari plugin to automatically count lung organoids from microscopy imaging data. A Faster R-CNN model was trained on patches of microscopy data. Model inference is run using a sliding window approach, with a 50% overlap and the option for predictiing on multiple window sizes and scales, the results of which are then merged using NMS.

![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/main/readme-content/demo-plugin.gif)

## Intended Audience & Supported Data

This plugin has been developed and tested with 2D CZI microscopy images of lunch organoids. The images had been previously converted from a 3D stack to 2D using an extended focus algorithm. This plugin may be used as a baseline for developers who wish to extend the plugin to work with other types of input images and/or improve the detection algorithm. 

## Dependencies

```napari-organoid-counter``` uses the ```napari-aicsimageio```<sup>[1]</sup> plugin for reading and processing CZI images.

[1] AICSImageIO Contributors (2021). AICSImageIO: Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python [Computer software]. GitHub. https://github.com/AllenCellModeling/aicsimageio

## Quickstart

The use of the napari-organoid-counter plugin is straightforward. Here is a step-to-step guide of the standard workflow:
1. You can load the image or images you wish to process into the napari viewer with a simple drag and drop
2. You can then select the layer you wish to work on by the drop-down box at the top of the input configurations
3. To improve the way the image is visualised you can pre-process them by clicking the _Preprocess_ button and the image layer will automatically be updated with the result
4. If you have a Faster R-CNN model you wish to use for the prediction, you can browser and select this by clicking on the _Choose_ button. Otherwise, the default model will be automatically downloaded from [here](https://zenodo.org/record/7708763#.ZDe6pS8Rpqs). Note that your own model must follow the specifications described here _(TODO)_.
5. You can adjust the _Window sizes_ and _Downsampling_ parameters to define the window size in the sliding window inference and the downsampling that is performed on the window. If you have multiple objects with different sizes, it might be good to set multiple window sizes, with corresponding downsampling rates. You can seperate these with a comma in the text box (e.g. ```2048, 512```). After you have set _Window sizes_ and _Downsampling_ hit **Enter** for the chanegs to be accepted.
6. By clicking the _Run Organoid Counter_ button the detection algorithm will run and a new shapes layer will be added to the viewer, with bounding boxes are placed around the detected organoid. You can add, edit or remove boxes using the _layer controls_ window (top left). The _Number of detected organoids_ will show you the number of organoids in the layer in real time. You can switch between viewing the box ids and model confidence for each box by toggling the _display text_ box in the _layer controls_ window. Boxes added by the user will by default have a confidence of 1.
7. If you feel that your model is over- or under-predicting you can use the _Model confidence_ scroll bar and select the value which best suits your problem. Default confidence is set to 0.8.
8. If you objects are typically bigger or smaller than those displayed you can use the _Minimum Diameter_ slider to set the minimum diameter of your objects. Default value is 30 um.
9. After you are happy with the detection results (and your manual corrections if any), you can save the  bounding boxes as a json file by clicking _Save boxes_. 

The _Take Screenshot_ button has the same functionality as _File -> Save screenshot_. The _Reset Configs_ button will reset the image and all parameters to the original settings. To save your results, first select the shapes layer you wish to save form the dropdown menu. To save features of the detected organoids (diameters when approximating organoid as an ellipse and organoid area) in a csv file click _Save features_. To save the bounding boxes as a json file click _Save boxes_.


## Getting Help

If you encounter any problems, please [file an issue] along with a detailed description.

## How to Cite
If you use this plugin for your work, please cite it using the following:
```
@software{christina_bukas_2022_6457904,
  author       = {Christina Bukas},
  title        = {{HelmholtzAI-Consultants-Munich/napari-organoid- 
                   counter: first release of napari plugin for lung
                   organoid counting}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.1.0-beta},
  doi          = {10.5281/zenodo.6457904},
  url          = {https://doi.org/10.5281/zenodo.6457904}
}
```



