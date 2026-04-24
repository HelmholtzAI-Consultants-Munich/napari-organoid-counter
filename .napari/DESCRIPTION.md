## Description

A napari plugin to automatically count lung organoids from microscopy imaging data. Several deep learning (DL) models were trained on patches of 2D microscopy data—some for object detection only, and one for both detection and binary classification. Model inference is run using a sliding window approach, with a 50% overlap and the option for predicting on multiple window sizes and scales, the results of which are then merged using NMS.

## Fearures of latest version
Here is a list of the main features that napari-organoid-counter offers:

Support for multiple DL models: 

* **Object Detection Only (DO)** - pretrained models: Faster R-CNN (DO), YOLOv3 (DO), SSD (DO), and RTMDet (DO). The data used for training these models along with the code for training can be found [here](https://www.kaggle.com/datasets/christinabukas/mutliorg).
<!-- * **Detection and Binary Classification (BC)** - pretrained models: Currently, YOLOv3 (BC) is supported, which not only detects organoids but also differentiates between two types of organoids (Class 0 and Class 1). Class 0 organoids are represented with Green and Class 1 with Blue. Bounding boxes for low confidence predictions will remain in Magenta. -->
* **Pyramid Model Inference** – Run inference using a sliding window approach, with tunable parameters for window size and downsampling rate.
* **Model Confidence** – A new tunable parameter to adjust and refine predictions based on confidence levels.
* **Annotation of Up to 10 Classes** – You can now annotate up to 10 different organoid classes, each with a unique color. Bounding box colors for each class can be adjusted using key bindings.
* **Saving Annotations with Class Labels** – When saving annotations, the class label is recorded in the .json file based on the bounding box color.
* **Load and Correct Existing Annotations** – You can now load and modify previously saved annotations (note: these must have been saved using v3 of the plugin).
<!-- * **Improved Readability** – Only model confidence is displayed in the viewer for better readability. -->
* **Data Browser** - Easily navigate thought all the images in your folder from the integrated data browser. The user can open the desired images by clickin on it and can save the annotations as final (visualized in <span style="color:green">green</span>) or as a draft for unfinished annotations (visualized in <span style="color:yellow">yellow</span>).
* **Preprocessing**: image preprocessing happens automatically when an image is loaded from the data browser. 
<!-- * **Interactive Workflow** – Work seamlessly with multiple shape layers, adjusting parameters and switching between layers as needed. -->

## Quickstart

After installation, you can start napari (either by typing ```napari``` in your terminal or by launching the application) and select the plugin from the drop down menu.

The use of the napari-organoid-counter plugin is straightforward. Here is a step-to-step guide of the standard workflow:

**1. Load the Data Folder**
* Set the folder you intend to work on from the Data Browser.

**2. Navigate through the Images**
* Click on the images shown in the Data Browser to load them.
* Use the Next Image button to load the next unannotated image in the folder.

**3. Model Selection**


* Pre-trained models: The plugin provides the following models, automatically downloaded if needed.

* **Detection Only (DO):** Faster R-CNN (DO), YOLOv3 (DO). These models will be automatically downloaded from Zenodo. Predicted bounding boxes will appear in default color magenta.

<!-- * **Binary Classification (BC):** YOLOv3 (BC). This model is also automatically downloaded from Zenodo. It predicts bounding boxes and classifies organoids as Green (Class 0) or Blue (Class 1), while low confidence cases remain in Magenta. -->

**4.  Annotation Mode Selection**
* You can select the annotation mode that best suits your needs. The annotation mode will determine the number of classes you can assign to the objects in the image. 
<!-- **4. Annotation Mode & Model Constraints**
* If a DO model is selected, all annotations modes are available.
* If a BC model is selected, the DO annotation mode is disabled. If you attempt to run the detection by clicking _Run Organoid Counter_, the following error message will appear. -->

<!-- ![Error Model and Annotation Mode Mismatch](/readme-content/warning_BC_model_DO_annotation.png) -->

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

![Key Bindings](/readme-content/key-bindings.png)

* When selecting an annotation mode, an information pop-up will appear, displaying the key bindings associated with that specific mode.

![Annotation Mode Info](/readme-content/annotation_mode_info.png)

* If using an annotation mode (other than DO), you must assign the correct colors before saving. If any bounding boxes are missing a valid class color, a warning appears.

![Valid Colors Warning](/readme-content/valid_colors_warning.png)

**8. Adjusting Confidence & Object Size Thresholds**
* If you feel that your model is over- or under-predicting you can use the _Model confidence_ scroll bar and select the value which best suits your problem. Default confidence is set to 0.8.
* If you objects are typically bigger or smaller than those displayed you can use the _Minimum Diameter_ slider to set the minimum diameter of your objects. Default value is 30 um.

**9. Saving Results**
* After you are happy with the detection results (and your manual corrections if any), you can decide how to save your results. 
* _Save as draft_: if you are not finished with your annotations but want to save your progress, you can click on _Save as draft_. This will save the annotations in a .json file, which can be loaded again and further edited. Annotations that don't have all valid classes can still be saved as draft.
*  _Save_: if you are finished with your annotations, you can click on _Save_. This will save the annotations in a .json file and the boxes features (size, area, class) in a .csv file. The annotationscan be anyway loaded again and further edited.
* The status of the annotations (draft or final) is visualized in the data browser with different colors: white for unannotated, <span style="color:yellow">yellow</span> for draft and <span style="color:green">green</span> for final:

![Data Browser Status](/readme-content/Data_browser_window.png)

**Remark**: When opening a new images the current progress is automatically saved by the tool.

**10. Additional Options**
* _Take Screenshot_ button has the same functionality as _File -> Save screenshot_. 
* _Set as Default_ button will save the current settings (model param., size and confidence thresholds) as default for the next time you open the plugin.
* _Reset_ button will reset the settings to the original values. 

## Getting Help

If you encounter any problems, please [file an issue](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/issues) along with a detailed description.

## Intended Audience & Supported Data

This plugin has been developed and tested with 2D CZI microscopy images of lung organoids. The images have been previously converted from a 3D stack to 2D using an extended focus algorithm. This plugin only supports single channel grayscale images. This plugin may be used as a baseline for developers who wish to extend the plugin to work with other types of input images and/or improve the detection algorithm. 
