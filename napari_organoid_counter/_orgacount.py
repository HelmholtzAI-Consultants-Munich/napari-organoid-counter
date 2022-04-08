import math
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import block_reduce
from skimage.feature import canny
from skimage.measure import regionprops,label
from skimage.morphology import remove_small_objects, erosion, disk
from skimage.transform import resize
#import cv2

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

def apply_normalization(img):
    img = np.squeeze(img) #self.viewer.layers[self.image_layer_name].data)
    img = img.astype(np.float64)
    #Normalise and return img to range 0-255
    img_min = np.min(img) # 31.3125 png 0
    img_max = np.max(img) # 2899.25 png 178
    img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    return img_norm

def count_organoids(img_original, img_scale, downsampling, min_diameter_um, sigma, low_threshold=10, high_threshold=25, background_intensity=40):
    img = block_reduce(img_original, block_size=(downsampling, downsampling), func=np.mean)   
    img_scale = [downsampling*x for x in img_scale] # update resolutions
    # get mask of well and background
    mask = np.where(img<background_intensity,False,True)
    # find edges in image
    edges = canny(image=img,
                sigma=sigma,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                mask=mask)
    
    edges = ndi.binary_dilation(edges) # dilate edges
    filled = ndi.binary_fill_holes(edges) # fill holes
    filled = erosion(filled, disk(2))
    labels = label(filled)
    region = regionprops(labels)
    # remove objects larger than 30% of the image
    for prop in region:
        if prop.area > 0.3*img.shape[0]*img.shape[1]:
            filled[prop.coords] = 0
    # get min organoid size in pixels
    min_radius_um = min_diameter_um//2
    min_area = math.pi * min_radius_um**2 
    min_area_pix = min_area / (img_scale[0] * img_scale[1])
    min_size_pix = round(min_area_pix)
    filled = remove_small_objects(filled, min_size_pix)
    segmentation = label(filled)
    segmentation = resize(segmentation, (img_original.shape[0], img_original.shape[1]),
                        preserve_range=True,
                        order=0,
                        anti_aliasing=True)
    segmentation = segmentation.astype(np.uint32) # convert to int32 in case there are more than 255 detected objects
    return segmentation

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

'''
def add_text_to_img(img, organoid_number, downsampling=1):
    """
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
    """
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
'''