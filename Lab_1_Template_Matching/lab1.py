import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: image preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    ###Your code here###
    ###
    img_gray = 0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]
    return img_gray.astype(np.uint8)


def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    

    ###Your code here####
    ###
    def convolve(img, kernel): # only supports 3x3 kernel
        # flip kernel
        kernel = np.flip(kernel)
        
        # pad image
        padded_img = np.zeros((img.shape[0]+2, img.shape[1]+2))
        padded_img[1:-1,1:-1] = img
        
        # construct vectorized image and kernel
        img_h, img_w = img.shape
        ker_h, ker_w = kernel.shape
        h_indices = np.repeat(np.arange(img_h), img_w) + np.repeat(np.arange(ker_h), ker_w).reshape(-1,1)
        w_indices = np.tile(np.arange(img_w), img_h) + np.tile(np.arange(ker_w), ker_h).reshape(-1,1)
        vectorized_img = padded_img[h_indices, w_indices]
        vectorized_kernel = kernel.reshape(1, -1)
        
        # perform convolution as matrix multiplication
        vectorized_convolution = vectorized_kernel@vectorized_img
        
        # unvectorize result
        convolution = vectorized_convolution.reshape(img.shape)
        
        return convolution
        
    img_grad_h = convolve(img, sobelh)
    img_grad_v = convolve(img, sobelv)
    img_grad_d1 = convolve(img, sobeld1)
    img_grad_d2 = convolve(img, sobeld2)
    
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    ###Your code here###
    ###
    img_pad = img_pad.astype(img.dtype)
    img_pad[pad_height_bef:-pad_height_aft,pad_width_bef:-pad_width_aft] = img
    return img_pad




##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 6 nested for-loops. 
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    img = img.astype('float')
    template = template.astype('float')
    
    C = template.shape[2]
    output = np.zeros((Ho, Wo))
    for h_o in range(Ho):
        for w_o in range(Wo):
            numerator = 0
            window_mag_squared = 0
            kernel_mag_squared = 0
            for h_k in range(Hk):
                for w_k in range(Wk):
                    for c in range(C):
                        p_i = img[h_o+h_k, w_o+w_k, c]
                        p_k = template[h_k, w_k, c]
                        
                        numerator += p_i*p_k
                        window_mag_squared += p_i**2
                        kernel_mag_squared += p_k**2
                        
            denominator = np.sqrt(kernel_mag_squared*window_mag_squared)
            output[h_o, w_o] = numerator/denominator
                        
    return output


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    img = img.astype('float')
    template = template.astype('float')
    
    output = np.zeros((Ho, Wo))
    kernel_mag_squared = np.sum(np.square(template))
    for h_o in range(Ho):
        for w_o in range(Wo):
            window = img[h_o:h_o+Hk, w_o:w_o+Wk]
            numerator = np.sum(window * template)
            
            window_mag_squared = np.sum(np.square(window))
            denominator = np.sqrt(kernel_mag_squared*window_mag_squared)
            
            output[h_o, w_o] = numerator/denominator
    
    return output




def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    img = img.astype('float')
    template = template.astype('float')
    
    # construct channel indices
    C = template.shape[2]
    c_indices = np.repeat(np.arange(C), Hk*Wk).reshape(1,-1)
    
    # construct img indices
    win_h_indices = np.repeat(np.arange(Ho), Wo)
    ker_h_indices = np.repeat(np.arange(Hk), Wk)
    img_h_indices = np.tile(win_h_indices.reshape(-1,1) + ker_h_indices, (1, C))
    
    win_w_indices = np.tile(np.arange(Wo), Ho)
    ker_w_indices = np.tile(np.arange(Wk), Hk)
    img_w_indices = np.tile(win_w_indices.reshape(-1,1) + ker_w_indices, (1, C))
    
    # construct img vector
    vectorized_img = img[img_h_indices, img_w_indices, c_indices]
    
    # construct template indices
    tpl_h_indices = np.tile(ker_h_indices, (1, C))
    tpl_w_indices = np.tile(ker_w_indices, (1, C))
    
    # construct template vector
    vectorized_template = template[tpl_h_indices, tpl_w_indices, c_indices].reshape(-1, 1)
    
    # perform convolution as matric multiplication
    vectorized_ccorr = vectorized_img@vectorized_template
    numerators = vectorized_ccorr.reshape(Ho, Wo)
    
    # compute normalization term
    vectorized_mag_template = np.ones_like(template).reshape(-1, 1)
    vectorized_img_squared = np.square(vectorized_img)
    windows_mag_squared = (vectorized_img_squared@vectorized_mag_template).reshape(Ho, Wo)
    kernel_mag_squared = np.sum(np.square(template))
    denominators = np.sqrt(kernel_mag_squared*windows_mag_squared)
    
    output = numerators/denominators
    
    return output


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    ###Your code here###
    ###
    output = np.zeros_like(response)
    
    # thresholding
    if threshold is not None:
        response[response < threshold] = 0
    
    # suppression
    while np.any(response):
        max_loc = np.unravel_index(np.argmax(response), response.shape)
        output[max_loc] = 1
        response[
            max(0, max_loc[0]-suppress_range[0]):min(response.shape[0], max_loc[0]+suppress_range[0]),
            max(0, max_loc[1]-suppress_range[1]):min(response.shape[1], max_loc[1]+suppress_range[1])] = 0
    
    return output

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    img = img.astype('float')
    template = template.astype('float')
    
    output = np.zeros((Ho, Wo))
    template_mean = np.mean(template, (0,1)).reshape(1,1,-1)
    kernel_mag_squared = np.sum(np.square(template-template_mean))
    for h_o in range(Ho):
        for w_o in range(Wo):
            window = img[h_o:h_o+Hk, w_o:w_o+Wk]
            
            window_mean = np.mean(window, (0,1)).reshape(1,1,-1)
            numerator = np.sum((window-window_mean) * (template-template_mean))
            
            window_mag_squared = np.sum(np.square(window-window_mean))
            denominator = np.sqrt(kernel_mag_squared*window_mag_squared)
            
            output[h_o, w_o] = numerator/denominator
            
    return output






###############################################
"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_squares(response, img_ori=None, rec_shape=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()
    H, W = response.shape[:2]
    if rec_shape is None:
        h_rec, w_rec = 25, 25
    else:
        h_rec, w_rec = rec_shape

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.rectangle(response, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (255, 0, 0), 2)
        if img_ori is not None:
            img_ori = cv2.rectangle(img_ori, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (0, 255, 0), 2)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)
