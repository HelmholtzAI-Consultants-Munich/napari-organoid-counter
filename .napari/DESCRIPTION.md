## Description

A napari plugin to automatically count lung organoids from microscopy imaging data. Several deep learning (DL) models were trained on patches of 2D microscopy data—some for object detection only, and one for both detection and binary classification. Model inference is run using a sliding window approach, with a 50% overlap and the option for predicting on multiple window sizes and scales, the results of which are then merged using NMS.

## What's new in v3?
Here is a list of the main changes v3 of napari-organoid-counter offers:

Support for multiple DL models: 

* **Object Detection Only (DO)** - pretrained models: Faster R-CNN (DO), YOLOv3 (DO), SSD (DO), and RTMDet (DO). The data used for training these models along with the code for training can be found [here](https://www.kaggle.com/datasets/christinabukas/mutliorg).

* **Detection and Binary Classification (BC)** - pretrained models: Currently, YOLOv3 (BC) is supported, which not only detects organoids but also differentiates between two types of organoids (Class 0 and Class 1). Class 0 organoids are represented with Green and Class 1 with Blue. Bounding boxes for low confidence predictions will remain in Magenta.

* **Pyramid Model Inference** – Run inference using a sliding window approach, with tunable parameters for window size and downsampling rate.
* **Model Confidence** – A new tunable parameter to adjust and refine predictions based on confidence levels.
* **Annotation of Up to 10 Classes** – You can now annotate up to 10 different organoid classes, each with a unique color. Bounding box colors for each class can be adjusted using key bindings (see Quickstart instructions for details).
* **Saving Annotations with Class Labels** – When saving annotations, the class label is recorded in the .json file based on the bounding box color.
* **Load and Correct Existing Annotations** – You can now load and modify previously saved annotations (note: these must have been saved using v3 of the plugin).
* **Improved Readability** – Only model confidence is displayed in the viewer for better readability.
* **Bug Fixed** – Fixed an issue where box thickness varied at different downsampling rates.
* **Interactive Workflow** – Work seamlessly with multiple shape layers, adjusting parameters and switching between layers as needed.

Technical Extensions:
* Allows for Python 3.10
* Extensive testing

## Installation

This plugin has been tested with python 3.9 and 3.10 - you may consider using conda to create your dedicated environment before running the `napari-organoid-counter`.

1. You can install `napari-organoid-counter` via [pip](https://pypi.org/project/napari-organoid-counter/):

    ```pip install napari-organoid-counter```

   To install latest development version :

    ```pip install git+https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter.git```

2. Additionally, you will then need to install one additional dependency: 

  ``` mim install "mmcv<2.2.0,>=2.0.0rc4" ```

**Note:** mmcv requires **Microsoft Visual Studio 2022 Build Tools**. Download and install Visual Studio Build Tools [here] (https://visualstudio.microsoft.com/es/visual-cpp-build-tools/) and make sure to select **"Desktop development with C++"** option (https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/ten_classes_annotation/readme-content/visual_studio_build_tools.png)

For installing on a Windows machine directly from within napari, follow the instuctions [here](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/ten_classes_annotation/readme-content/How%20to%20install%20on%20a%20Windows%20machine.pdf). Step 2 additionally needs to be performed here too (mim install "mmcv<2.2.0,>=2.0.0rc4").

## Quickstart

After installation, you can start napari (either by typing ```napari``` in your terminal or by launching the application) and select the plugin from the drop down menu.

The use of the napari-organoid-counter plugin is straightforward. Here is a step-to-step guide of the standard workflow:

**1. Load and Select an Image**
* Drag and drop the image(s) you wish to process into the napari viewer.
* Select the layer you want to work on using the drop-down box at the top of the input configurations.

**2. Preprocessing (Optional)**
* To improve the way the image is visualised you can pre-process them by clicking the _Preprocess_ button and the image layer will automatically be updated with the result.

**3. Model Selection**

You can use either:
* Your own model: If you have a Faster R-CNN model you wish to use for the prediction, you can browser and select this by clicking on the _Choose_ button. Note that your own model must follow the specifications described here _(TODO)_.
* Pre-trained models: The plugin provides the following models, automatically downloaded if needed.

* **Detection Only (DO):** Faster R-CNN (DO), YOLOv3 (DO), SSD (DO), RTMDet (DO). These models will be automatically downloaded from Zenodo. Predicted bounding boxes will appear in default color magenta.

* **Binary Classification (BC):** YOLOv3 (BC). This model is also automatically downloaded from Zenodo. It predicts bounding boxes and classifies organoids as Green (Class 0) or Blue (Class 1), while low confidence cases remain in Magenta.

**4. Annotation Mode & Model Constraints**
* If a DO model is selected, all annotations modes are available.
* If a BC model is selected, the DO annotation mode is disabled. If you attempt to run the detection by clicking _Run Organoid Counter_, the following error message will appear.

![Error Model and Annotation Mode Mismatch](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/ten_classes_annotation/readme-content/warning_BC_model_DO_annotation.png)

**5. Adjust Model Parameters**
You can adjust the _Window sizes_ and _Downsampling_ parameters to define the window size in the sliding window inference and the downsampling that is performed on the image. If you have multiple objects with different sizes, it might be good to set multiple window sizes, with corresponding downsampling rates. You can seperate these with a comma in the text box (e.g. ```2048, 512```). After you have set _Window sizes_ and _Downsampling_ hit **Enter** for each for the changes to be accepted. 
* **_Downsampling parameter:_** To detect large organoids (and ignore smaller structures) you may need to increase the downsampling rate, whereas if your organoids are small and are being missed by the algorithm, consider reducing the downsampling rate. 
* **_Window size parameter:_** The window size can also impact the number of objects detected: typically a ratio of 512 to 1 between window size and downsampling rate would give optimal results, while larger window sizes would lead to a drop in performance. However, please note that small window sizes will signicantly impact the runtime of the algorithm.

**6. Running the Organoid Counter**

By clicking the _Run Organoid Counter_ button the detection algorithm will run and a new shapes layer will be added to the viewer, with bounding boxes are placed around the detected organoid. 

**7. Correcting & Annotating Bounding Boxes**
* You can add, edit or remove boxes using the _layer controls_ window (top left). Make sure the _Select vertices_ tool is clicked to select bounding boxes. 
* Multiple bounding boxes can be selected at the same time holding _Shift_ and clicking on the bounding boxes. 
* The _Number of detected organoids_ will show you the number of organoids in the layer in real time. 
* You can switch between viewing the model confidence for each box by toggling the _display text_ box in the _layer controls_ window. Boxes added by the user will by default have a confidence of 1.
* You can choose between different annotation modes and annotate up to 10 classes, based on your preferences. To change the color of the bounding boxes, the following key bindings are assigned for each specific class. 

![Key Bindings](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/ten_classes_annotation/readme-content/key-bindings.png)

* When selecting an annotation mode, an information pop-up will appear, displaying the key bindings associated with that specific mode.

![Annotation Mode Info](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/ten_classes_annotation/readme-content/annotation_mode_info.png)

* If using an annotation mode (other than DO), you must assign the correct colors before saving. If any bounding boxes are missing a valid class color, a warning appears.

![Valid Colors Warning](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/ten_classes_annotation/readme-content/valid_colors_warning.png)

**8. Adjusting Confidence & Object Size Thresholds**
* If you feel that your model is over- or under-predicting you can use the _Model confidence_ scroll bar and select the value which best suits your problem. Default confidence is set to 0.8.
* If you objects are typically bigger or smaller than those displayed you can use the _Minimum Diameter_ slider to set the minimum diameter of your objects. Default value is 30 um.

**9. Saving Results**
* After you are happy with the detection results (and your manual corrections if any), you can select which shapes layer results to save via the dropdown box. 
* _Save boxes_: saves the bounding box coordinates (along with the box id, model confidence for that box, x and y scale for that image, and class labels) as a json file. You can select where to save your file. The class is indicated by the color of the bounding box. In DO mode, class is 'null'. 
*  _Save features_: saves the features in a csv file. The saved features will include the organoid id, the two lengths of the boundig box (which corresponds to the min. and max. diameter of the organoid, approximated as an ellipse), the area of the organoid (again approximated as an ellipse).

**10. Additional Options**
* _Take Screenshot_ button has the same functionality as _File -> Save screenshot_. 
* _Reset Configs_ button will reset the image and all parameters to the original settings. 

## Getting Help

If you encounter any problems, please [file an issue](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/issues) along with a detailed description.

## Intended Audience & Supported Data

This plugin has been developed and tested with 2D CZI microscopy images of lung organoids. The images have been previously converted from a 3D stack to 2D using an extended focus algorithm. This plugin only supports single channel grayscale images. This plugin may be used as a baseline for developers who wish to extend the plugin to work with other types of input images and/or improve the detection algorithm. 

## Dependencies

```napari-organoid-counter``` uses the ```napari-aicsimageio```<sup>[1]</sup> <sup>[2]</sup> plugin for reading and processing CZI images.

[1] Eva Maxfield Brown, Dan Toloudis, Jamie Sherman, Madison Swain-Bowden, Talley Lambert, AICSImageIO Contributors (2021). AICSImageIO: Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python [Computer software]. GitHub. https://github.com/AllenCellModeling/aicsimageio

[2] Eva Maxfield Brown, Talley Lambert, Peter Sobolewski, Napari-AICSImageIO Contributors (2021). Napari-AICSImageIO: Image Reading in Napari using AICSImageIO [Computer software]. GitHub. https://github.com/AllenCellModeling/napari-aicsimageio

The latest version also uses models developed with the ```mmdetection``` package <sup>[3]</sup>, see [here](https://github.com/open-mmlab/mmdetection)

[3] Chen, Kai, et al. "MMDetection: Open mmlab detection toolbox and benchmark." arXiv preprint arXiv:1906.07155 (2019).


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


