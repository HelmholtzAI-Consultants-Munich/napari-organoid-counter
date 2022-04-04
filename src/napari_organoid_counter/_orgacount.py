import os
import math
import numpy as np
from aicsimageio import AICSImage
from scipy import ndimage as ndi
from skimage.measure import block_reduce
from skimage.feature import canny
from skimage.measure import regionprops,label
from skimage.morphology import remove_small_objects, erosion, disk
import cv2

def add_text_to_img(img, organoid_number, downsampling=1):
    '''
    Adds the number of organoids detected as text to the image and returns it
    Parameters
    ----------
    img: numpy array
        The image on which text needs to be added
    organoid_number: int
        The number of organoids detected - to be added as text 
    downsampling: int
        the downsampling of the image will affect the text size
    Returns
    -------
    img: numpy array
        The image with text added to it
    
    '''
    # define thickness and font size of the text depending on the downsampling rate
    thickness=2
    if downsampling==1: 
        fontSize = 10 #6
        thickness = 12 #4
    elif downsampling<4: fontSize = 3
    else: fontSize = 2
    # add text to image
    img = cv2.putText(img, 
        'Organoids: '+str(organoid_number), 
        org=(round(img.shape[1]*0.05), round(img.shape[0]*0.1)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontSize,
        thickness=thickness,
        color=(255))
    return img

def setup_bboxes(segmentation): 
    '''
    Given segmentation of organoids set up bounding boxes to visualise in napari
    Parameters
    ----------
    segmentation: numpy array
        The segmentation of organoids
    Returns
    -------
    bbox_properties: dict
        Holds the diameters of each detected object
    text_parameters: dict
        Holds parameters for visualisation of bounding boxes and parameters
    bboxes: numpy array
        An array where each entry is a bounding box in form [minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]
    
    '''
    bboxes = []
    diameter1 = []
    diameter2 = []
    # for each detected object compute bounding box and min and max diameter
    for region in regionprops(segmentation):
        minr, minc, maxr, maxc = region.bbox
        bbox_rect = np.array([[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]])
        bboxes.append(bbox_rect)
        diameter1.append(maxr-minr)
        diameter2.append(maxc-minc)
    # set up properties of bounding box and text with parameters for napari visualisation
    bbox_properties = {'diameter1': diameter1, 'diameter2': diameter2}
    text_parameters = {
        'text': 'D1: {diameter1}\nD2: {diameter2}',
        'anchor': 'upper_left',
        'translation': [-5, 0],
        'size': 8,
        'color': 'green',
    }
    bboxes = np.array(bboxes)
    return bbox_properties, text_parameters, bboxes

def circle_area(r):
    '''Compute and return the area of a circle of radius r'''
    return math.pi * r**2 

def compute_real_values(self, d1, d2): 
    '''
    Given diameters of the organoid approximates as an ellipse compute the diameters and area of the organoids in um
    Parameters
    ----------
    d1: int
        The  first diameter of the organoid in pixels - organoids are approximated as ellipses
    d2: int
        The  second diameter of the organoid in pixels - organoids are approximated as ellipses
    Returns
    -------
    d1: float
        The  first diameter of the organoid in um
    d2: float
        The  second diameter of the organoid in um
    area: float
        The area of the organoid in um - approximated as an ellipse
    '''
    # d1 and d2 are already in original resolution
    d1 = d1 * self.img_resX_orig
    d2 = d2 * self.img_resY_orig
    area = math.pi * d1 * d2
    return d1, d2, area
    


