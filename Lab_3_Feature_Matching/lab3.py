import numpy as np
from skimage import filters
from skimage import feature
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
import math

### REMOVE THIS
from cv2 import findHomography

from utils import pad, unpad

import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

##################### PART 1 ###################

# 1.1 IMPLEMENT
def harris_corners(img, window_size=3, k=0.04):
    '''
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    '''

    H, W= img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))

    # YOUR CODE HERE

    I_x = filters.sobel_h(img)
    I_y = filters.sobel_v(img)

    A = convolve(np.square(I_x), window, mode='constant')
    B = convolve(I_x*I_y, window, mode='constant')
    C = convolve(np.square(I_y), window, mode='constant')

    det = A*C - B*B
    trace = A + C

    response = det - k*np.square(trace)

    # END        
    return response

# 1.2 IMPLEMENT
def naive_descriptor(patch):
    '''
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    '''
    feature = []
    ### YOUR CODE HERE

    feature = ((patch - np.mean(patch)) / np.std(patch)).flatten()

    ### END YOUR CODE

    return feature

# GIVEN
def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    '''
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (x, y) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    '''

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[np.max([0,y-(patch_size//2)]):y+((patch_size+1)//2),
                      np.max([0,x-(patch_size//2)]):x+((patch_size+1)//2)]
      
        desc.append(desc_func(patch))
   
    return np.array(desc)

# GIVEN
def make_gaussian_kernel(ksize, sigma):
    '''
    Good old Gaussian kernel.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    yy, xx = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(yy) + np.square(xx)) / np.square(sigma))

    return kernel / kernel.sum()


# 1.2 IMPLEMENT
def simple_sift(patch):
    '''
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Use the gradient orientation to determine the bin, and the gradient magnitude * weight from
    the Gaussian kernel as vote weight.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    '''
    
    # You can change the parameter sigma, which has been default to 3
    weights = np.flipud(np.fliplr(make_gaussian_kernel(patch.shape[0],3)))
    
    histogram = np.zeros((4,4,8))
    
    # YOUR CODE HERE

    grad_x = filters.sobel_h(patch)
    grad_y = filters.sobel_v(patch)

    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad_mag_weighted = grad_mag * weights
    grad_orientation = np.remainder((np.arctan2(grad_x, grad_y) * 180 / np.pi) + 360, 360)

    for y in range(4):
        for x in range(4):
            hist = np.histogram(grad_orientation[y*4:(y+1)*4, x*4:(x+1)*4], bins=8, range=(0, 360), weights=grad_mag_weighted[y*4:(y+1)*4, x*4:(x+1)*4])[0]
            histogram[y, x] = hist
    
    feature = histogram.flatten() / np.linalg.norm(histogram)
  
    # END
    return feature

# 1.3 IMPLEMENT
def top_k_matches(desc1, desc2, k=2):
    '''
    Compute the Euclidean distance between each descriptor in desc1 versus all descriptors in desc2 (Hint: use cdist).
    For each descriptor Di in desc1, pick out k nearest descriptors from desc2, as well as the distances themselves.
    Example of an output of this function:
    
        [(0, [(18, 0.11414082134194799), (28, 0.139670625444803)]),
         (1, [(2, 0.14780585099287238), (9, 0.15420019834435536)]),
         (2, [(64, 0.12429203239414029), (267, 0.1395765079352806)]),
         ...<truncated>
    '''
    match_pairs = []
    
    # YOUR CODE HERE

    dist = cdist(desc1, desc2, 'euclidean')
    dist_argsort = np.argsort(dist, axis=1)
    
    for i in range(dist.shape[0]):
        nearest_idxs = dist_argsort[i, :k]
        nearest_dist = dist[i, nearest_idxs]
        match_pairs.append((i, list(zip(nearest_idxs, nearest_dist))))
  
    # END
    return match_pairs

# 1.3 IMPLEMENT
def ratio_test_match(desc1, desc2, match_threshold):
    '''
    Match two set of descriptors using the ratio test.
    Output should be a numpy array of shape (k,2), where k is the number of matches found. 
    In the following sample output:
        array([[  3,   0],
               [  5,  30],
               [ 11,   9],
               [ 18,   7],
               [ 24,   5],
               [ 30,  17],
               [ 32,  24],
               [ 46,  23], ... <truncated>
              )
              
        desc1[3] is matched with desc2[0], desc1[5] is matched with desc2[30], and so on.
    
    All other match functions will return in the same format as does this one.
    
    '''
    match_pairs = []
    top_2_matches = top_k_matches(desc1, desc2)
    # YOUR CODE HERE

    for i, i_2_matches in top_2_matches:
        if i_2_matches[0][1] / i_2_matches[1][1] < match_threshold:
            match_pairs.append([i, i_2_matches[0][0]])
   
    # END
    # Modify this line as you wish
    match_pairs = np.array(match_pairs)
    return match_pairs

# GIVEN
def compute_cv2_descriptor(im, method=cv2.SIFT_create()):
    '''
    Detects and computes keypoints using one of the implementations in OpenCV
    You can use:
        cv2.SIFT_create()

    Do note that the keypoints coordinate is (col, row)-(x,y) in OpenCV. We have changed it to (row,col)-(y,x) for you. (Consistent with out coordinate choice)
    '''
    kpts, descs = method.detectAndCompute(im, None)
    
    keypoints = np.array([(kp.pt[1],kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])
    
    return keypoints, descs, angles, sizes

##################### PART 2 ###################

# GIVEN
def transform_homography(src, h_matrix, getNormalized = True):
    '''
    Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    '''
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed

# 2.1 IMPLEMENT
def compute_homography(src, dst):
    '''
    Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    '''
    h_matrix = np.eye(3, dtype=np.float64)
  
    # YOUR CODE HERE
    def to_homo(arr):
        return np.concatenate((arr, np.ones((arr.shape[0], 1))), axis=1)

    # construct normalization matrix for src
    src_mean = np.mean(src, axis=0)
    src_scale = np.sqrt(2) / np.mean(np.sqrt(np.sum(np.square(src-src_mean), axis=1)))
    src_T = np.array([
                [src_scale, 0, 0],
                [0, src_scale, 0],
                [-src_scale*src_mean[0], -src_scale*src_mean[1], 1]
            ])
    
    # construct normalization matrix for dst
    dst_mean = np.mean(dst, axis=0)
    dst_scale = np.sqrt(2) / np.mean(np.sqrt(np.sum(np.square(dst-dst_mean), axis=1)))
    dst_T = np.array([
                [dst_scale, 0, 0],
                [0, dst_scale, 0],
                [-dst_scale*dst_mean[0], -dst_scale*dst_mean[1], 1]
            ])

    # convert to homogeneous coordinates
    src_homo = to_homo(src)
    dst_homo = to_homo(dst)
    
    # normalize src
    src_normal = src_homo @ src_T

    # normalize dst
    dst_normal = dst_homo @ dst_T

    # construct matrix A
    A = np.zeros((2*src.shape[0], 9), dtype=np.float64)
    A[0::2,:3] = -src_normal
    A[0::2,6:] = src_normal*dst_normal[:,0,np.newaxis]
    A[1::2,3:6] = -src_normal
    A[1::2,6:] = src_normal*dst_normal[:,1,np.newaxis]
    
    # compute homography H
    _, _, v_T = np.linalg.svd(A)
    H_normal = v_T.T[:, -1].reshape(3, 3)
    H = np.linalg.inv(dst_T.T) @ H_normal @ src_T.T

    h_matrix = H

    # END 

    return h_matrix

# 2.2 IMPLEMENT
def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, delta=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        sampling_ratio: percentage of points selected at each iteration
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]

    max_inliers = np.zeros(N, dtype=int)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    def to_homo(arr):
        return np.concatenate((arr, np.ones((arr.shape[0], 1))), axis=1)
    
    def from_homo(arr):
        return arr[:,:2] / arr[:,2,np.newaxis]

    for _ in range(n_iters):

        # Take random sample
        sample = np.random.choice(N, size=n_samples, replace=False)
        matched1_sample = matched1_unpad[sample]
        matched2_sample = matched2_unpad[sample]

        # Compute homography
        H = compute_homography(matched1_sample, matched2_sample)

        # Count inliers
        matched1_unpad_proj = from_homo(to_homo(matched1_unpad) @ H.T)
        matched2_unpad_proj = from_homo(to_homo(matched2_unpad) @ np.linalg.inv(H.T))
        distances1 = np.linalg.norm(matched2_unpad-matched1_unpad_proj, axis=1)
        distances2 = np.linalg.norm(matched1_unpad-matched2_unpad_proj, axis=1)
        inliers = np.nonzero(np.logical_and(distances1<delta, distances2<delta))[0]

        # Store largest number of inliers
        if inliers.shape[0] > n_inliers:
            max_inliers = inliers
            n_inliers = inliers.shape[0]
    
    # Recompute homography based on largest number of inliers
    H = compute_homography(matched1_unpad[max_inliers], matched2_unpad[max_inliers])
 
    ### END YOUR CODE
    return H, matches[max_inliers]

##################### PART 3 ###################
# GIVEN FROM PREV LAB
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters)

    Also include the array of values corresponding to the bins, in descending order.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res

# GIVEN
def angle_with_x_axis(pi, pj):  
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    y, x = pi[0]-pj[0], pi[1]-pj[1] 
    
    if x == 0:
        return np.pi/2  
    
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

# GIVEN
def midpoint(pi, pj):
    '''
    Get y and x coordinates of the midpoint of I and J
    '''
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2

# GIVEN
def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    y,x = pi[0]-pj[0], pi[1]-pj[1] 
    return np.sqrt(x**2+y**2)

# 3.1 IMPLEMENT
def shift_sift_descriptor(desc):
    '''
       Generate a virtual mirror descriptor for a given descriptor.
       Note that you have to shift the bins within a mini histogram, and the mini histograms themselves.
       e.g:
       Descriptor for a keypoint
       (the dimension is (128,), but here we reshape it to (16,8). Each length-8 array is a mini histogram.)
      [[  0.,   0.,   0.,   5.,  41.,   0.,   0.,   0.],
       [ 22.,   2.,   1.,  24., 167.,   0.,   0.,   1.],
       [167.,   3.,   1.,   4.,  29.,   0.,   0.,  12.],
       [ 50.,   0.,   0.,   0.,   0.,   0.,   0.,   4.],
       
       [  0.,   0.,   0.,   4.,  67.,   0.,   0.,   0.],
       [ 35.,   2.,   0.,  25., 167.,   1.,   0.,   1.],
       [167.,   4.,   0.,   4.,  32.,   0.,   0.,   5.],
       [ 65.,   0.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  74.,   1.,   0.,   0.],
       [ 36.,   2.,   0.,   5., 167.,   7.,   0.,   4.],
       [167.,  10.,   0.,   1.,  30.,   1.,   0.,  13.],
       [ 60.,   2.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  54.,   3.,   0.,   0.],
       [ 23.,   6.,   0.,   4., 167.,   9.,   0.,   0.],
       [167.,  40.,   0.,   2.,  30.,   1.,   0.,   0.],
       [ 51.,   8.,   0.,   0.,   0.,   0.,   0.,   0.]]
     ======================================================
       Descriptor for the same keypoint, flipped over the vertical axis
      [[  0.,   0.,   0.,   3.,  54.,   0.,   0.,   0.],
       [ 23.,   0.,   0.,   9., 167.,   4.,   0.,   6.],
       [167.,   0.,   0.,   1.,  30.,   2.,   0.,  40.],
       [ 51.,   0.,   0.,   0.,   0.,   0.,   0.,   8.],
       
       [  0.,   0.,   0.,   1.,  74.,   0.,   0.,   0.],
       [ 36.,   4.,   0.,   7., 167.,   5.,   0.,   2.],
       [167.,  13.,   0.,   1.,  30.,   1.,   0.,  10.],
       [ 60.,   1.,   0.,   0.,   0.,   0.,   0.,   2.],
       
       [  0.,   0.,   0.,   0.,  67.,   4.,   0.,   0.],
       [ 35.,   1.,   0.,   1., 167.,  25.,   0.,   2.],
       [167.,   5.,   0.,   0.,  32.,   4.,   0.,   4.],
       [ 65.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       
       [  0.,   0.,   0.,   0.,  41.,   5.,   0.,   0.],
       [ 22.,   1.,   0.,   0., 167.,  24.,   1.,   2.],
       [167.,  12.,   0.,   0.,  29.,   4.,   1.,   3.],
       [ 50.,   4.,   0.,   0.,   0.,   0.,   0.,   0.]]
    '''
    # YOUR CODE HERE

    desc_16_8 = desc.reshape(16, 8)
    row_indices = np.tile(np.concatenate((np.arange(12, 16), np.arange(8, 12), np.arange(4, 8), np.arange(0, 4))).reshape(-1, 1), (1, 8))
    col_indices = np.tile(np.roll(np.arange(7, -1, -1), 1).reshape(1, -1), (16, 1))
    res_16_8 = desc_16_8[row_indices, col_indices]
    res = res_16_8.flatten()
   
    # END
    return res

# 3.1 IMPLEMENT
def create_mirror_descriptors(img):
    '''
    Return the output for compute_cv2_descriptor (which you can find in utils.py)
    Also return the set of virtual mirror descriptors.
    Make sure the virtual descriptors correspond to the original set of descriptors.
    '''
    # YOUR CODE HERE

    kps, descs, angles, sizes = compute_cv2_descriptor(img)
    mir_descs = np.apply_along_axis(shift_sift_descriptor, 1, descs)
 
    # END
    return kps, descs, sizes, angles, mir_descs

# 3.2 IMPLEMENT
def match_mirror_descriptors(descs, mirror_descs, threshold = 0.7):
    '''
    First use `top_k_matches` to find the nearest 3 matches for each keypoint. Then eliminate the mirror descriptor that comes 
    from the same keypoint. Perform ratio test on the two matches left. If no descriptor is eliminated, perform the ratio test 
    on the best 2. 
    '''
    three_matches = top_k_matches(descs, mirror_descs, k=3)

    match_result = []
    # YOUR CODE HERE

    for i, i_3_matches in three_matches:
        i_3_matches_without_i = [match for match in i_3_matches if match[0] != i]
        if i_3_matches_without_i[0][1] / i_3_matches_without_i[1][1] < threshold:
            match_result.append([i, i_3_matches_without_i[0][0]])
   
    match_result = np.array(match_result)

    # END
    return match_result

# 3.3 IMPLEMENT
def find_symmetry_lines(matches, kps):
    '''
    For each pair of matched keypoints, use the keypoint coordinates to compute a candidate symmetry line.
    Assume the points associated with the original descriptor set to be I's, and the points associated with the mirror descriptor set to be
    J's.
    '''
    rhos = []
    thetas = []
    # YOUR CODE HERE

    matched_kps_src = np.array(list(zip(kps[matches[:, 0]][:,0], kps[matches[:, 0]][:,1])), dtype="i,i")
    matched_kps_dst = np.array(list(zip(kps[matches[:, 1]][:,0], kps[matches[:, 1]][:,1])), dtype="i,i")
    matched_kps = np.stack((matched_kps_src, matched_kps_dst), axis=1)

    rhos = np.apply_along_axis(lambda kp: midpoint(kp[0], kp[1]), 1, matched_kps)
    thetas = np.apply_along_axis(lambda kp: angle_with_x_axis(kp[0], kp[1]), 1, matched_kps)

    # END
    
    return rhos, thetas

# 3.4 IMPLEMENT
def hough_vote_mirror(matches, kps, im_shape, window=1, threshold=0.5, num_lines=1):
    '''
    Hough Voting:
                 0<=thetas<= 2pi      , interval size = 1 degree
        -diagonal <= rhos <= diagonal , interval size = 1 pixel
    Feel free to vary the interval size.
    '''
    rhos, thetas = find_symmetry_lines(matches, kps)
    
    # YOUR CODE HERE

    # specify min, max
    max_rho = int(np.hypot(*im_shape[:2]))
    min_rho = -max_rho
    min_theta, max_theta = 0, 2*np.pi

    # specify interval
    interval_rho = 1
    interval_theta = np.pi/180

    # specify dimension
    dim_rho = int((max_rho-min_rho)/interval_rho)
    dim_theta = int((max_theta-min_theta)/interval_theta)

    # construct accumulator array
    A = np.zeros((dim_rho, dim_theta), dtype=int)
    dists = np.linspace(min_rho, max_rho, num=dim_rho, endpoint=False)
    thets = np.linspace(min_theta, max_theta, num=dim_theta, endpoint=False)

    # performs hough voting
    theta_votes = ((thetas-min_theta)/interval_theta).astype(int)
    rho_votes = (((rhos[:, 1]*np.cos(thetas) + rhos[:, 0]*np.sin(thetas))-min_rho)/interval_rho).astype(int)
    votes = np.array(list(zip(rho_votes, theta_votes)), dtype="i,i")
    uniq_votes, count = np.unique(votes, return_counts=True)
    uniq_votes_rho, uniq_votes_theta = zip(*uniq_votes)
    A[uniq_votes_rho, uniq_votes_theta] = count

    # find peak bins
    _, rho_values, theta_values = find_peak_params(A, [dists,thets], window, threshold)
    rho_values = rho_values[:num_lines]
    theta_values = theta_values[:num_lines]
  
    # END
    
    return rho_values, theta_values

##################### PART 4 ###################

# 4.1 IMPLEMENT
def match_with_self(descs, kps, threshold=0.8):
    '''
    Use `top_k_matches` to match a set of descriptors against itself and find the best 3 matches for each descriptor.
    Discard the trivial match for each trio (if exists), and perform the ratio test on the two matches left (or best two if no match is removed)
    '''
   
    matches = []
    
    # YOUR CODE HERE

    three_matches = top_k_matches(descs, descs, k=3)

    for i, i_3_matches in three_matches:
        i_3_matches_without_i = [match for match in i_3_matches if match[0] != i]
        if i_3_matches_without_i[0][1] / i_3_matches_without_i[1][1] < threshold:
            matches.append([i, i_3_matches_without_i[0][0]])
   
    matches = np.array(matches)
   
    # END
    return matches

# 4.2 IMPLEMENT
def find_rotation_centers(matches, kps, angles, sizes, im_shape):
    '''
    For each pair of matched keypoints (using `match_with_self`), compute the coordinates of the center of rotation and vote weight. 
    For each pair (kp1, kp2), use kp1 as point I, and kp2 as point J. The center of rotation is such that if we pivot point I about it,
    the orientation line at point I will end up coinciding with that at point J. 
    
    You may want to draw out a simple case to visualize first.
    
    If a candidate center lies out of bound, ignore it.
    '''
    # Y-coordinates, X-coordinates, and the vote weights 
    Y = []
    X = []
    W = []
    
    # YOUR CODE HERE

    # convert angles to radians
    angles = angles*np.pi/180

    # get matched kps, angles, sizes
    matched_kps = zip(kps[matches[:,0]], kps[matches[:,1]])
    matched_angles = zip(angles[matches[:,0]], angles[matches[:,1]])
    matched_sizes = zip(sizes[matches[:,0]], sizes[matches[:,1]])

    # calculate angle diffs
    angle_diffs_1 = np.remainder(angles[matches[:,0]]-angles[matches[:,1]]+2*np.pi, 2*np.pi)
    angle_diffs_2 = np.remainder(angles[matches[:,1]]-angles[matches[:,0]]+2*np.pi, 2*np.pi)
    angle_diffs = np.minimum(angle_diffs_1, angle_diffs_2)

    for matched_kp, matched_angle, matched_size, angle_diff in zip(matched_kps, matched_angles, matched_sizes, angle_diffs):
        if angle_diff < np.pi/180:
            continue

        d = distance(matched_kp[0], matched_kp[1])
        gamma = angle_with_x_axis(matched_kp[0], matched_kp[1])

        beta = (matched_angle[0] - matched_angle[1] + np.pi) / 2
        r = (d * np.sqrt(1 + np.square(np.tan(beta)))) / 2

        x_c = matched_kp[0][1] + r*np.cos(beta+gamma)
        y_c = matched_kp[0][0] + r*np.sin(beta+gamma)

        if x_c < 0 or x_c >= im_shape[1] or y_c < 0 or y_c >= im_shape[0]:
            continue

        w_c = np.square(np.exp(-abs(matched_size[0]-matched_size[1])/(matched_size[0]+matched_size[1])))

        X.append(x_c)
        Y.append(y_c)
        W.append(w_c)

    X = np.array(X)
    Y = np.array(Y)
    W = np.array(W)

    # END
    
    return Y,X,W

# 4.3 IMPLEMENT
def hough_vote_rotation(matches, kps, angles, sizes, im_shape, window=1, threshold=0.5, num_centers=1):
    '''
    Hough Voting:
        X: bound by width of image
        Y: bound by height of image
    Return the y-coordianate and x-coordinate values for the centers (limit by the num_centers)
    '''
    
    Y,X,W = find_rotation_centers(matches, kps, angles, sizes, im_shape)
    
    # YOUR CODE HERE

    # specify min, max
    min_x, max_x = 0, im_shape[1]
    min_y, max_y = 0, im_shape[0]

    # specify interval
    interval_x = 1
    interval_y = 1

    # specify dimension
    dim_x = int((max_x-min_x)/interval_x)
    dim_y = int((max_y-min_y)/interval_y)

    # construct accumulator array
    A = np.zeros((dim_x, dim_y))
    xs = np.linspace(min_x, max_x, num=dim_x, endpoint=False).astype(int)
    ys = np.linspace(min_y, max_y, num=dim_y, endpoint=False).astype(int)

    # performs hough voting
    x_votes = ((X-min_x)/interval_x).astype(int)
    y_votes = ((Y-min_y)/interval_y).astype(int)
    for x_vote, y_vote, w_vote in zip(x_votes, y_votes, W):
        A[x_vote, y_vote] += w_vote

    # find peak bins
    votes, x_values, y_values = find_peak_params(A, [xs,ys], window, threshold)
    x_values = x_values[:num_centers]
    y_values = y_values[:num_centers]

    # print(votes)

    # END
    
    return y_values, x_values
