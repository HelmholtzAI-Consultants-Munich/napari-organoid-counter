## Description

A napari plugin to automatically count lung organoids from microscopy imaging data. The original implementation can be found in the [Organoid-Counting](https://github.com/HelmholtzAI-Consultants-Munich/Organoid-Counting) repository, which has been adapted here to work as a napari plugin. The CannyEdgeDetection algorithm is used for detecting the organoids and pre-processing steps ahve been added, specific to this type of data to obtain optimal resutls.

![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/main/readme-content/demo-plugin.gif)

## Intended Audience & Supported Data

This plugin has been developed and tested with 2D CZI microscopy images of lunch organoids. The images had been previously converted from a 3D stack to 2D using an extended focus algorithm. This plugin may be used as a baseline for developers who wish to extend the plugin to work with other types of input images and/or improve the detection algorithm. 

## Quickstart

The use of the napari-organoid-counter plugin is straightforward. After loading the image or images you wish to process into the napari viewer, you must first pre-process them by clicking the _Preprocess_ button and the image layer will automatically be updated with the result. Next, you can adjust any of the parameters used in the algorithm (downsamppling, minimum organoid diamtere and sigma, i.e. kernel sixe for the Cannny Edge Detection algorithm) by using the corresponding sliders. By clicking the _Run Organoid Counter_ button the detection algorithm will run and a new shapes layer will be added to the viewer, with bounding boxes are placed around the detected organoid. You can add, edit or remove boxes using the _layer controls_ window and update the _Number of detected organoids_ displayed by clicking the _Update Number_ button. 

The _Take Screenshot_ button has the same functionality as _File -> Save screenshot_. The _Reset Configs_ button will reset the image and all parameters to the original settings. To save your results, first select the shapes layer you wish to save form the dropdown menu. To save features of the detected organoids (diameters when approximating organoid as an ellipse and organoid area) in a csv file click _Save features_. To save the bounding boxes as a json file click _Save boxes_.


## Getting Help

If you encounter any problems, please [file an issue] along with a detailed description.

## How to Cite




