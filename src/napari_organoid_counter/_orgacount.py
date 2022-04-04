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

def apply_normalization(img):
    '''Normalise and return img to range 0-255'''
    img_min = np.min(img) # 31.3125 png 0
    img_max = np.max(img) # 2899.25 png 178
    img_norm = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
    return img_norm

class OrgaCount():
    '''
    The OrgaCount class is used for segemnting lung organoids from brightfiled microscopy images.
    This class includes  functions:
    - '_update_resolutions' function, which updates the resolution in x and y given the downsampling size
    - 'get_current_downsampling' function, which returns the current downsampling size
    - 'update_min_organoid_size' function, which updates the minimun radius in um of organoids
    - 'update_donwnsampling' function, which updates the downsampling size and resolution of image
    - 'update_sigma' function, which updates the sigma used in the Canny Edge Detection algorithm
    - 'update_low_threshold' function, which updates the low threshold used in the Canny Edge Detection algorithm
    - 'update_high_threshold' function, which updates the high threshold used in the Canny Edge Detection algorithm
    - '_min_pixel_area' function, which computes the minimum organoid area in pixels (approximate as circle)
    - 'compute_real_values' function, which computes the diameters and area of the organoids in um (approx. ellipse)
    - 'apply_morphologies' function, which detects the organoids in the image and returns the segmentation
    Parameters
    ----------
        root_path : str
            The directory in which image we wish to segment is stored
        img_path : str
            The name of the image we wish to segment. The image must be in czi format
        downsampling_size : int
            The downsamplign applied to the image as a pre-processign step
        min_diameter_um : int
            The minimum diameter of the organoids in um defined by the user (organoids are approximated as circles)
        sigma : int
            The sigma used in the Canny Edge Detection algorithm
        low_threshold : int
            The low hysteresis threshold used in the Canny Edge Detection algorithm
        high_threshold : int 
            The high hysteresis threshold used in the Canny Edge Detection algorithm 
    Attributes
    -------
        img_resX_orig : float
            The resultion in x axis of the original image read from disk
        img_resY_orig : float
            The resultion in y axis of the original image read from disk
        img_original : np.array
            The original image read from disk after removing dims of one
        img_orig_norm: np.array
            The original image normalised
        downsampling_size : int
            See parameters
        sigma : int
            See parameters
        low_threshold : int
            See parameters
        high_threshold : int
            See parameters
        background_intensity : int
            The intensity of the background of the image, around the plate
        min_radius_um : int
            The minimum radius of the organoids in um (organoids are approximated as circles)
        img_resX: float
            The resolution of the image in x axis after applying current downsampling
        img_resY: float
            The resolution of the image in y axis after applying current downsampling
    '''
    def __init__(self, 
                root_path,
                img_path, 
                downsampling_size, 
                min_diameter_um, 
                sigma, 
                low_threshold, 
                high_threshold,
                background_intensity):
        '''
        img_czi = AICSImage(os.path.join(root_path, img_path))
        self.img_resX_orig = img_czi.physical_pixel_sizes.X # in micrometers
        self.img_resY_orig = img_czi.physical_pixel_sizes.Y
        self.img_original = np.squeeze(img_czi.data)
        self.img_original = self.img_original.astype(np.float64)
        self.img_orig_norm = apply_normalization(self.img_original)
        print('Opened image: ', img_path, 'with shape: ', self.img_original.shape)
        '''
        self.downsampling_size = downsampling_size
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.background_intensity = background_intensity
        self.min_radius_um = min_diameter_um//2 #15 # min diameter defined by collaborators as d=30 micrometers. Min area A=pi*r^2

    def _update_resolutions(self):
        ''' Update the resolution in x and y given the downsampling size'''
        self.img_resX = self.downsampling_size * self.img_resX_orig
        self.img_resY = self.downsampling_size * self.img_resY_orig
    
    def get_current_downsampling(self):
        ''' Return the current downsampling size'''
        return self.downsampling_size

    def update_min_organoid_size(self, min_diameter_size):
        ''' Update the minimun radius in um of organoids'''
        self.min_radius_um = min_diameter_size//2

    def update_donwnsampling(self, new_size):
        ''' Update the downsampling size and resolution of image'''
        self.downsampling_size = new_size
        self._update_resolutions()

    def update_sigma(self, new_sigma):
        ''' Update the sigma used in the Canny Edge Detection algorithm'''
        self.sigma = new_sigma

    def update_low_threshold(self, new_low):
        ''' Update the low threshold used in the Canny Edge Detection algorithm'''
        self.low_threshold = new_low

    def update_high_threshold(self, new_high):
        ''' Update the high threshold used in the Canny Edge Detection algorithm'''
        self.high_threshold = new_high

    def _min_pixel_area(self):
        ''' Compute the minimum organoid area in pixels (approximate as circle) using the minimum radius of organoids 
        defined by the user'''
        min_area = circle_area(self.min_radius_um)
        min_area_pix = min_area / (self.img_resX * self.img_resY)
        return round(min_area_pix)
    
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
    
    def apply_morphologies(self):
        '''
        Detect the organoids in the image and return the segmentation
        Returns
        -------
        segmentation: numpy array
            The resulting segmentation of the image
        '''
        # downsample 
        img = block_reduce(self.img_original, block_size=(self.downsampling_size, self.downsampling_size), func=np.mean)   
        print('Image resampled to size: ', img.shape)
        self._update_resolutions()
        # normalise
        img = apply_normalization(img)
        # get mask of well and background
        mask = np.where(img<self.background_intensity,False,True)
        # find edges in image
        edges = canny(
                    image=img,
                    sigma=self.sigma,
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold,
                    mask = mask)
        # dilate edges
        edges = ndi.binary_dilation(edges)
        # fill holes
        filled = ndi.binary_fill_holes(edges)
        filled = erosion(filled, disk(2))
        labels = label(filled)
        region = regionprops(labels)
        # remove objects larger than 30% of the image
        for prop in region:
            if prop.area > 0.3*img.shape[0]*img.shape[1]:
                filled[prop.coords] = 0
        # get min organoid size in pixels
        min_size_pix = self._min_pixel_area()
        filled = remove_small_objects(filled, min_size_pix)
        segmentation = label(filled)
        return segmentation