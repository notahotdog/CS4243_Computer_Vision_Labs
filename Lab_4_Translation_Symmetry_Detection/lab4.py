import cv2
import numpy as np
import math
from numpy.ma.core import filled
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise

### Part 1

def detect_points(img, min_distance, rou, pt_num, patch_size, tau_rou, gamma_rou):
    """
    Patchwise Shi-Tomasi point extraction.

    Hints:
    (1) You may find the function cv2.goodFeaturesToTrack helpful. The initial default parameter setting is given in the notebook.

    Args:
        img: Input RGB image. 
        min_distance: Minimum possible Euclidean distance between the returned corners. A parameter of cv2.goodFeaturesToTrack
        rou: Parameter characterizing the minimal accepted quality of image corners. A parameter of cv2.goodFeaturesToTrack
        pt_num: Maximum number of corners to return. A parameter of cv2.goodFeaturesToTrack
        patch_size: Size of each patch. The image is divided into several patches of shape (patch_size, patch_size). There are ((h / patch_size) * (w / patch_size)) patches in total given a image of (h x w)
        tau_rou: If rou falls below this threshold, stops keypoint detection for that patch
        gamma_rou: Decay rou by a factor of gamma_rou to detect more points.
    Returns:
        pts: Detected points of shape (N, 2), where N is the number of detected points. Each point is saved as the order of (height-corrdinate, width-corrdinate)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, c = img.shape

    Np = pt_num * 0.9 # The required number of keypoints for each patch. `pt_num` is used as a parameter, while `Np` is used as a stopping criterion.

    # YOUR CODE HERE

    kpts_list = []

    h_tops = np.arange(0, h, patch_size)
    w_lefts = np.arange(0, w, patch_size)

    for h_top in h_tops:
        for w_left in w_lefts:
            h_bottom = min(h_top+patch_size, h)
            w_right = min(w_left+patch_size, w)

            patch = img_gray[h_top:h_bottom, w_left:w_right]

            while True:
                kpts = cv2.goodFeaturesToTrack(patch, pt_num, rou, min_distance).reshape((-1, 2))
                kpts += np.array([[h_top, w_left]])
                rou *= gamma_rou
                if len(kpts) > Np or rou < tau_rou:
                    break
            
            kpts_list.append(kpts)
    
    pts = np.concatenate(kpts_list)

    # END

    return pts


def detect_points_whole(img, min_distance, rou, pt_num, tau_rou, gamma_rou):
    """
    Patchwise Shi-Tomasi point extraction.

    Hints:
    (1) You may find the function cv2.goodFeaturesToTrack helpful. The initial default parameter setting is given in the notebook.

    Args:
        img: Input RGB image. 
        min_distance: Minimum possible Euclidean distance between the returned corners. A parameter of cv2.goodFeaturesToTrack
        rou: Parameter characterizing the minimal accepted quality of image corners. A parameter of cv2.goodFeaturesToTrack
        pt_num: Maximum number of corners to return. A parameter of cv2.goodFeaturesToTrack
        patch_size: Size of each patch. The image is divided into several patches of shape (patch_size, patch_size). There are ((h / patch_size) * (w / patch_size)) patches in total given a image of (h x w)
        tau_rou: If rou falls below this threshold, stops keypoint detection for that patch
        gamma_rou: Decay rou by a factor of gamma_rou to detect more points.
    Returns:
        pts: Detected points of shape (N, 2), where N is the number of detected points. Each point is saved as the order of (height-corrdinate, width-corrdinate)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img_gray.shape)
    h, w, c = img.shape

    Np = pt_num * 0.9 # The required number of keypoints for each patch. `pt_num` is used as a parameter, while `Np` is used as a stopping criterion.

    # YOUR CODE HERE
    while True:
        kpts = cv2.goodFeaturesToTrack(img_gray, pt_num, rou, min_distance).reshape((-1, 2))
        rou *= gamma_rou
        if len(kpts) > Np or rou < tau_rou:
            break
    
    pts = kpts
    print(rou)

    # END

    return pts


def extract_point_features(img, pts, window_patch):
    """
    Extract patch feature for each point.

    The patch feature for a point is defined as the patch extracted with this point as the center.

    Note that the returned pts is a subset of the input pts. 
    We discard some of the points as they are close to the boundary of the image and we cannot extract a full patch.

    Please normalize the patch features by subtracting the mean intensity and dividing by its standard deviation.

    Args:
        img: Input RGB image.
        pts: Detected point corners from detect_points().
        window_patch: The window size of patch cropped around the point. The final patch is of size (5 + 1 + 5, 5 + 1 + 5) = (11, 11). The center is the given point.
                      For example, suppose an image is of size (300, 400). The point is located at (50, 60). The window size is 5. 
                      Then, we use the cropped patch, i.e., img[50-5:50+5+1, 60-5:60+5+1], as the feature for that point. The patch size is (11, 11), so the dimension is 11x11=121.
    Returns:
        pts: A subset of the input points. We can extract a full patch for each of these points.
        features: Patch features of the points of the shape (N, (window_patch*2 + 1)^2), where N is the number of points
    """


    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE

    pts_list = []
    features_list = []

    for pt in pts:
        pt = pt.astype(int)
        top = pt[0]-window_patch
        bottom = pt[0]+window_patch+1
        left = pt[1]-window_patch
        right = pt[1]+window_patch+1

        if top < 0 or bottom > h or left < 0 or right > w:
            continue

        pts_list.append(pt)
        features_list.append(img_gray[top:bottom, left:right].reshape(-1))
    
    pts = np.array(pts_list)
    features = np.array(features_list)

    # Normalize features
    eps = 1e-9 # to prevent division by zero
    features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + eps)

    # End

    return pts, features

def mean_shift_clustering(features, bandwidth):
    """
    Mean-Shift Clustering.

    There are various ways of implementing mean-shift clustering. 
    The provided default bandwidth value may not be optimal to your implementation.
    Please fine-tune the bandwidth so that it can give the best result.

    Args:
        img: Input RGB image.
        bandwidth: If the distance between a point and a clustering mean is below bandwidth, this point probably belongs to this cluster.
    Returns:
        clustering: A dictionary, which contains three keys as follows:
                    1. cluster_centers_: a numpy ndarrary of shape [N_c, 2] for the naive point cloud task and [N_c, 121] for the main task (patch features of the point).
                                         Each row is the center of that cluster.
                    2. labels_:  a numpy nadarray of shape [N,], where N is the number of features. 
                                 labels_[i] denotes the label of the i-th feature. The label is between [0, N_c - 1]
                    3. bandwidth: bandwith value
    """
    # YOUR CODE HERE

    # Own implementation

    ## Find cluster centers

    stop_threshold = 1e-3 * bandwidth
    centers = []
    centers_n_neighs = []

    for feat in features:

        # Perform mean shift until convergence
        centroid = feat
        while True:
            distances = np.linalg.norm(features-centroid, axis=1)
            neighbors = np.nonzero(distances < bandwidth)[0]
            old_centroid = centroid
            centroid = np.mean(features[neighbors], axis=0)

            if np.linalg.norm(old_centroid-centroid) < stop_threshold:
                break
        
        # Keep cluster centers that has neighboring points in it's bandwidth
        if len(neighbors) > 0:
            centers.append(centroid)
            centers_n_neighs.append(len(neighbors))
    

    ## Remove duplicate centers
    # A duplicate is a center whose distance is less than bandwidth from another center.
    
    # Sort centers according to number of it's neighbors
    sorted_centers = np.array(centers)[np.argsort(np.array(centers_n_neighs))[::-1]]

    # Remove duplicate
    uniq = np.full((len(centers)), True)
    for i, center in enumerate(sorted_centers):
        if uniq[i]:
            distances = np.linalg.norm(sorted_centers-center, axis=1)
            neighbors = np.nonzero(distances < bandwidth)[0]
            
            uniq[neighbors] = False
            uniq[i] = True
    
    cluster_centers = sorted_centers[uniq]
    
    ## Assign labels
    from scipy.spatial.distance import cdist
    labels = np.argsort(cdist(features, cluster_centers), axis=1)[...,0]

    clustering = {
        "cluster_centers_": cluster_centers,
        "labels_": labels,
        "bandwidth": bandwidth
    }
    # END

    return clustering

def cluster(img, pts, features, bandwidth, tau1, tau2, gamma_h):
    """
    Group points with similar appearance, then refine the groups.

    "gamma_h" provides another way of fine-tuning bandwidth to avoid the number of clusters becoming too large.
    Alternatively, you can ignore "gamma_h" and fine-tune bandwidth by yourself.

    Args:
        img: Input RGB image.
        pts: Output from `extract_point_features`.
        features: Patch feature of points. Output from `extract_point_features`.
        bandwidth: Window size of the mean-shift clustering. In pdf, the bandwidth is represented as "h", but we use "bandwidth" to avoid the confusion with the image height
        tau1: Discard clusters with less than tau1 points
        tau2: Perform further clustering for clusters with more than tau2 points using K-means
        gamma_h: To avoid the number of clusters becoming too large, tune the bandwidth by gradually increasing the bandwidth by a factor gamma_h
    Returns:
        clusters: A list of clusters. Each cluster is a numpy ndarray of shape [N_cp, 2]. N_cp is the number of points of that cluster.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE

    from collections import defaultdict

    while True:
        clusters = []

        # Perform mean-shift clustering
        clustering = mean_shift_clustering(features, bandwidth)

        # Gather all clusters' features
        clusters_feats = defaultdict(list)
        clusters_pts = defaultdict(list)
        for i, label in enumerate(clustering["labels_"]):
            clusters_feats[label].append(features[i])
            clusters_pts[label].append(pts[i])
        
        # Post processing
        for i_c in clusters_feats.keys():
            cluster_feats = clusters_feats[i_c]
            cluster_pts = clusters_pts[i_c]
            n_members = len(cluster_feats)

            # Discard if members less than tau1
            if n_members < tau1:
                pass

            # Keep if members between tau1 and tau2
            elif n_members <= tau2:
                clusters.append(np.array(cluster_pts))

            # Partition when members more than tau2
            else: # len(cluster_pts) > tau2
                kmeans = KMeans(n_clusters=n_members//tau2).fit(cluster_feats)
                kmeans_pts = defaultdict(list)
                for i_k, kmeans_label in enumerate(kmeans.labels_):
                    kmeans_pts[kmeans_label].append(cluster_pts[i_k])

                for kmean_pts in kmeans_pts.values():
                    if len(kmean_pts) >= tau1:
                        clusters.append(np.array(kmean_pts))

        if len(clusters) <= pts.shape[0] / 3:
            break
        
        bandwidth *= gamma_h
    
    print(bandwidth)

    # END

    return clusters

### Part 2


def get_proposal(pts_cluster, tau_a, X):
    """
    Get the lattice proposal

    Hints:
    (1) As stated in the lab4.pdf, we give priority to points close to each other when we sample a triplet.
        This statement means that we can start from the three closest points and iterate N_a times.
        There is no need to go through every triplet combination.
        For instance, you can iterate over each point. For each point, you choose 2 of the 10 nearest points. The 3 points form a triplet.
        In this case N_a = num_points * 45.

    (2) It is recommended that you reorder the 3 points. 
        Since {a, b, c} are transformed onto {(0, 0), (1, 0), (0, 1)} respectively, the point a is expected to be the vertex opposite the longest side of the triangle formed by these three points

    (3) Another way of refining the choice of triplet is to keep the triplet whose angle (between the edges <a, b> and <a, c>) is within a certain range.
        The range, for instance, is between 20 degrees and 120 degrees.

    (4) You may find `cv2.getAffineTransform` helpful. However, be careful about the HW and WH ordering when you use this function.

    (5) If two triplets yield the same number of inliers, keep the one with closest 3 points.

    Args:
        pts_cluster: Points within the same cluster.
        tau_a: The threshold of the difference between the transformed corrdinate and integer positions.
               For example, if a point is transformed into (1.1, -2.03), the closest integer position is (1, -2), then the distance is sqrt(0.1^2 + 0.03^2) (for Euclidean distance case).
               If it is smaller than "tau_a", we consider this point as inlier.
        X: When we compute the inliers, we only consider X nearest points to the point "a". 
    Returns:
        proposal: A list of inliers. The first 3 inliers are {a, b, c}. 
                  Each inlier is a dictionary, with key of "pt_int" and "pt" representing the integer positions after affine transformation and orignal coordinates.
    """
    # YOU CODE HERE

    def to_homo(arr):
        return np.concatenate((arr, np.ones((arr.shape[0], 1))), axis=1)
    
    best_n_inlier = 0

    for pt1 in pts_cluster:
        closest_10 = pts_cluster[np.argsort(np.linalg.norm(pts_cluster-pt1, axis=1))][1:min(11, len(pts_cluster))]
        for i in range(min(10, len(pts_cluster)-1)):
            for j in range(i+1, min(10, len(pts_cluster)-1)):
                pt2 = closest_10[i]
                pt3 = closest_10[j]

                # Reorder to choose a, b, c
                sides = np.linalg.norm([pt2-pt3, pt1-pt3, pt1-pt2], axis=1)

                a, b, c = np.array([pt1, pt2, pt3])[np.argsort(sides)[::-1]]
                
                # Discard extreme angle
                ab_vector = (a-b) / np.linalg.norm(a-b)
                ac_vector = (a-c) / np.linalg.norm(a-c)
                angle = np.rad2deg(np.arccos(np.clip(np.dot(ab_vector, ac_vector), -1.0, 1.0)))
                if angle < 20 or angle > 120:
                    continue

                # Compute affine transform
                src = np.array([[a[1], a[0]], [b[1], b[0]], [c[1], c[0]]], dtype=np.float32)
                dst = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
                M = cv2.getAffineTransform(src, dst)

                # Count inliers
                pts_cluster_without_abc = np.array([pt for pt in pts_cluster if not(np.array_equal(pt, a) or np.array_equal(pt, b) or np.array_equal(pt, c))])
                closest_X = pts_cluster_without_abc[np.argsort(np.linalg.norm(pts_cluster_without_abc-a, axis=1))][:X-3]
                transformed = (M @ to_homo(np.fliplr(closest_X)).T).T
                rounded = np.rint(transformed).astype(int)
                distance_to_rounded = np.linalg.norm(transformed-rounded, axis=1)
                inlier_mask = distance_to_rounded < tau_a

                inliers = closest_X[inlier_mask]
                inliers_rounded = rounded[inlier_mask]

                if inliers.shape[0] >= best_n_inlier:
                    if inliers.shape[0] == best_n_inlier and best_n_inlier > 0:
                        best_inliers_dist = np.linalg.norm(best_inliers[0]-best_inliers[1]) + np.linalg.norm(best_inliers[0]-best_inliers[2])
                        cur_inliers_dist = np.linalg.norm(a-b) + np.linalg.norm(a-c)
                        if cur_inliers_dist >= best_inliers_dist:
                            continue
                    best_n_inlier = inliers.shape[0] + 3
                    best_inliers = np.concatenate(([a, b, c], inliers))
                    best_inliers_rounded = np.concatenate(([[0, 0], [1, 0], [0, 1]], inliers_rounded))
        
    proposal = []
    for i in range(best_n_inlier):
        proposal.append({"pt_int": best_inliers_rounded[i], "pt": best_inliers[i]})

    # END

    return proposal


def find_texels(img, proposal, texel_size=50):
    """
    Find texels from the given image.

    Hints:
    (1) This function works on RGB image, unlike previous functions such as point detection and clustering that operate on grayscale image.

    (2) You may find `cv2.getPerspectiveTransform` and `cv2.warpPerspective` helpful.
        Please refer to the demo in the notebook for the usage of the 2 functions.
        Be careful about the HW and WH ordering when you use this function.
    
    (3) As stated in the pdf, each texel is defined by 3 or 4 inlier keypoints on the corners.
        If you find this sentence difficult to understand, you can go to check the demo.
        In the demo, a corresponding texel is obtained from 3 points. The 4th point is predicted from the 3 points.


    Args:
        img: Input RGB image
        proposal: Outputs from get_proposal(). Proposal is a list of inliers.
        texel_size: The patch size (U, V) of the patch transformed from the quadrilateral. 
                    In this implementation, U is equal to V. (U = V = texel_size = 50.) The texel is a square.
    Returns:
        texels: A numpy ndarray of the shape (#texels, texel_size, texel_size, #channels).
    """
    # YOUR CODE HERE

    from itertools import combinations

    texels = []

    visited_quads = set()
    visited_tris = set()

    pts_dict = {tuple(inlier['pt_int']): i for i, inlier in enumerate(proposal)}

    for combo in combinations(range(len(proposal)), 3):

        pt1 = proposal[combo[0]]['pt_int']
        pt2 = proposal[combo[1]]['pt_int']
        pt3 = proposal[combo[2]]['pt_int']

        # Skip if triplets has been visited before
        if tuple(sorted([tuple(pt1.tolist()), tuple(pt2.tolist()), tuple(pt3.tolist())])) in visited_tris:
            continue
        else:
            visited_tris.add(tuple(sorted([tuple(pt1.tolist()), tuple(pt2.tolist()), tuple(pt3.tolist())])))

        sides = np.linalg.norm(np.array([
            pt2-pt3,
            pt1-pt3,
            pt1-pt2
            ]), axis=1)
        
        ideal_sides = np.linalg.norm(np.array([
            [1,0],
            [0,1],
            [1,-1]
            ]), axis=1)

        # Verify connected-ness
        if sorted(sides.tolist()) != sorted(ideal_sides.tolist()):
            continue
        
        # Infer 4th point
        i_a, i_b, i_c = np.array([combo[0], combo[1], combo[2]])[np.argsort(sides)[::-1]]
        a_int, b_int, c_int = (proposal[i]['pt_int'] for i in [i_a, i_b, i_c])
        a, b, c = (proposal[i]['pt'] for i in [i_a, i_b, i_c])
        d_int = a_int + (b_int - a_int) + (c_int - a_int)
        d = a + (b - a) + (c - a)

        # Check whether 4th point is also in proposal
        if tuple(d_int) in pts_dict:
            i_d = pts_dict[tuple(d_int)]
            # Skip if quadruplets has been visited before
            if tuple(sorted([i_a, i_b, i_c, i_d])) in visited_quads:
                continue
            else:
                visited_quads.add(tuple(sorted([i_a, i_b, i_c, i_d])))
        
        # Reorder the points as ([0, 0], [0, 1], [1, 0], [1, 1])
        order = np.argsort(np.array([tuple(a_int), tuple(b_int), tuple(c_int), tuple(d_int)], dtype="i,i"))
        a, b, c, d = np.array([tuple(a), tuple(b), tuple(c), tuple(d)], dtype="i,i")[order].tolist()

        # Warp texel
        src = np.fliplr(np.array([a, b, c, d], dtype=np.float32))
        dst = np.array([[0, 0], [0, texel_size], [texel_size, 0], [texel_size, texel_size]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        texel = cv2.warpPerspective(img, M, (texel_size, texel_size))

        texels.append(texel)
    
    texels = np.array(texels)

    # END
    return texels

def score_proposal(texels, a_score_count_min=3):
    """
    Calcualte A-Score.

    Hints:
    (1) Each channel is normalized separately.
        The A-score for a RGB texel is the average of 3 A-scores of each channel.

    (2) You can return 1000 (in our example) to denote a invalid A-score.
        An invalid A-score is usually results from clusters with less than "a_score_count_min" texels.

    Args:
        texels: A numpy ndarray of the shape (#texels, window, window, #channels).
        a_score_count_min: Minimal number of texels we need to calculate the A-score.
    Returns:
        a_score: A-score calculated from the texels. If there are no sufficient texels, return 1000.    
    """

    if texels.shape[0] == 0:
        return 1000

    K, U, V, C = texels.shape

    # YOUR CODE HERE

    # Normalize texels
    eps = 1e-9 # to prevent division by zero
    texels = (texels - np.mean(texels, axis=(1,2), keepdims=True)) / (np.std(texels, axis=(1,2), keepdims=True) + eps)

    if K < a_score_count_min:
        a_score = 1000
    else:
        a_score = np.mean(np.sum(np.std(texels, axis=0), axis=(0,1), keepdims=False) / (U*V*K))

    # END

    return a_score


### Part 3
# You are free to change the input argument of the functions in Part 3.
# GIVEN
def non_max_suppression(response, suppress_range, threshold=None):
    """
    Non-maximum Suppression for translation symmetry detection

    The general approach for non-maximum suppression is as follows:
        1. Perform thresholding on the input response map. Set the points whose values are less than the threshold as 0.
        2. Find the largest response value in the current response map
        3. Set all points in a certain range around this largest point to 0. 
        4. Save the current largest point
        5. Repeat the step from 2 to 4 until all points are set as 0. 
        6. Return the saved points are the local maximum.

    Args:
        response: numpy.ndarray, output from the normalized cross correlation
        suppress_range: a tuple of two ints (H_range, W_range). The points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    Returns:
        threshold: int, points with value less than the threshold are set to 0
    """
    H, W = response.shape[:2]
    H_range, W_range = suppress_range
    res = np.copy(response)

    if threshold is not None:
        res[res<threshold] = 0

    idx_max = res.reshape(-1).argmax()
    x, y = idx_max // W, idx_max % W
    point_set = set()
    while res[x, y] != 0:
        point_set.add((x, y))
        res[max(x - H_range, 0): min(x+H_range, H), max(y - W_range, 0):min(y+W_range, W)] = 0
        idx_max = res.reshape(-1).argmax()
        x, y = idx_max // W, idx_max % W
    for x, y in point_set:
        res[x, y] = response[x, y]
    return res

def template_match(img, proposal, threshold):
    """
    Perform template matching on the original input image.

    Hints:
    (1) You may find cv2.copyMakeBorder and cv2.matchTemplate helpful. The cv2.copyMakeBorder is used for padding.
        Alternatively, you can use your implementation in Lab 1 for template matching.

    (2) For non-maximum suppression, you can either use the one you implemented for lab 1 or the code given above.

    Returns:
        response: A sparse response map from non-maximum suppression. 
    """
    # YOUR CODE HERE

    a, b, c = (proposal[i]['pt'] for i in range(3))
    d = a + (b - a) + (c - a)

    top = min(a[0], b[0], c[0], d[0])
    bottom = max(a[0], b[0], c[0], d[0]) + 1
    left = min(a[1], b[1], c[1], d[1])
    right = max(a[1], b[1], c[1], d[1]) + 1

    template = img[top:bottom, left:right]

    pad_top, pad_bottom = template.shape[0] // 2 - (1 if template.shape[0] % 2 == 0 else 0), template.shape[0] // 2
    pad_left, pad_right = template.shape[1] // 2 - (1 if template.shape[1] % 2 == 0 else 0), template.shape[1] // 2
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, [0,0,0])

    res = cv2.matchTemplate(img_padded, template, cv2.TM_CCOEFF_NORMED)

    response = non_max_suppression(res, (int(template.shape[0] * 0.8), int(template.shape[1] * 0.8)), threshold=threshold)

    # END
    return response

def maxima2grid(img, proposal, response):
    """
    Estimate 4 lattice points from each local maxima.

    Hints:
    (1) We can transfer the 4 offsets between the center of the original template and 4 lattice unit points to new detected centers.

    Args:
        response: The response map from `template_match()`.

    Returns:
        points_grid: an numpy ndarray of shape (N, 2), where N is the number of grid points.
    
    """
    # YOUR CODE HERE

    a, b, c = (proposal[i]['pt'] for i in range(3))
    d = a + (b - a) + (c - a)

    corners = np.array([a, b, c, d])

    top, left = np.min(corners, axis=0)
    bottom, right = np.max(corners, axis=0)

    center = np.array([(top+bottom)//2, (left+right)//2])

    offsets = corners - center

    points_grid_list = []

    for max_center in np.argwhere(response):
        points_grid_list.append(offsets + max_center)
    
    points_grid = np.concatenate(points_grid_list)

    # END

    return points_grid


# def refine_grid(img, proposal, points_grid, merge_dist=40, quantile_thresh=0.01):
    """
    Refine the detected grid points.

    Args:
        points_grid: The output from the `maxima2grid()`.

    Returns:
        points: A numpy ndarray of shape (N, 2), where N is the number of refined grid points.
    """
    # YOUR CODE HERE
    from scipy.spatial.distance import cdist

    def merge_points(points_grid):
        # Merge close points
        merged_points = []
        # merge_dist = 20

        pairwise_dist = cdist(points_grid, points_grid)
        closest_neigh = np.argsort(pairwise_dist, axis=1)[:,1:]

        visited_j = set()
        for i, js in enumerate(closest_neigh):
            if i in visited_j:
                continue
            for j in js:
                if pairwise_dist[i,j] < merge_dist:
                    visited_j.add(j)
                    continue
                
                if j == js[0]:
                    merged_points.append(points_grid[i])
                else:
                    merged_points.append((points_grid[i] + points_grid[js[0]])//2)
                break
        
        return np.array(merged_points)
    
    def fill_points(merged_points):

        # quantile_thresh = 0.01

        # Fill in missing points
        pairwise_diff = merged_points[np.newaxis, :, :] - merged_points[:, np.newaxis, :]
        pairwise_angle = np.rad2deg(np.arctan2(pairwise_diff[...,0], (pairwise_diff[...,1])))

        # top
        top_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > 60, pairwise_angle < 120))]
        top_right_dist = np.linalg.norm(top_right_diff, axis=-1)
        nonzero_mask = top_right_dist > 0
        top_right_diff = top_right_diff[nonzero_mask]
        top_right_dist = top_right_dist[nonzero_mask]
        top_right_off = np.mean(top_right_diff[top_right_dist < np.quantile(top_right_dist, quantile_thresh)], axis=0)

        # left
        top_left_diff = pairwise_diff[np.nonzero(np.logical_or(pairwise_angle > 150, pairwise_angle < -150))]
        top_left_dist = np.linalg.norm(top_left_diff, axis=-1)
        nonzero_mask = top_left_dist > 0
        top_left_diff = top_left_diff[nonzero_mask]
        top_left_dist = top_left_dist[nonzero_mask]
        top_left_off = np.mean(top_left_diff[top_left_dist < np.quantile(top_left_dist, quantile_thresh)], axis=0)

        # bottom
        bottom_left_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > -120, pairwise_angle < -60))]
        bottom_left_dist = np.linalg.norm(bottom_left_diff, axis=-1)
        nonzero_mask = bottom_left_dist > 0
        bottom_left_diff = bottom_left_diff[nonzero_mask]
        bottom_left_dist = bottom_left_dist[nonzero_mask]
        bottom_left_off = np.mean(bottom_left_diff[bottom_left_dist < np.quantile(bottom_left_dist, quantile_thresh)], axis=0)

        # right
        bottom_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle < 30, pairwise_angle > -30))]
        bottom_right_dist = np.linalg.norm(bottom_right_diff, axis=-1)
        nonzero_mask = bottom_right_dist > 0
        bottom_right_diff = bottom_right_diff[nonzero_mask]
        bottom_right_dist = bottom_right_dist[nonzero_mask]
        bottom_right_off = np.mean(bottom_right_diff[bottom_right_dist < np.quantile(bottom_right_dist, quantile_thresh)], axis=0)

        filled_points = []
        for pt in merged_points:
            filled_points.extend([pt+top_right_off, pt+top_left_off, pt+bottom_left_off, pt+bottom_right_off])
        
        filled_points.extend(merged_points)

        return np.array(filled_points)
    
    merged_points = merge_points(points_grid)
    filled_points = fill_points(merged_points)
    points = np.array(merge_points(filled_points))

    # END

    return points

def refine_grid(img, proposal, points_grid):
    """
    Refine the detected grid points.

    Args:
        points_grid: The output from the `maxima2grid()`.

    Returns:
        points: A numpy ndarray of shape (N, 2), where N is the number of refined grid points.
    """
    # YOUR CODE HERE
    from scipy.spatial.distance import cdist

    def merge_points(points_grid):
        # Merge close points
        merged_points = []
        merge_dist = 20

        pairwise_dist = cdist(points_grid, points_grid)
        closest_neigh = np.argsort(pairwise_dist, axis=1)[:,1:]

        visited_j = set()
        for i, js in enumerate(closest_neigh):
            if i in visited_j:
                continue
            for j in js:
                if pairwise_dist[i,j] < merge_dist:
                    visited_j.add(j)
                    continue
                
                if j == js[0]:
                    merged_points.append(points_grid[i])
                else:
                    merged_points.append((points_grid[i] + points_grid[js[0]])//2)
                break
        
        return np.array(merged_points)
    
    def fill_points(merged_points):

        quantile_thresh = 0.01

        # Fill in missing points
        pairwise_diff = merged_points[np.newaxis, :, :] - merged_points[:, np.newaxis, :]
        pairwise_angle = np.rad2deg(np.arctan2(pairwise_diff[...,0], (pairwise_diff[...,1])))

        # top-right
        top_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > 30, pairwise_angle < 60))]
        top_right_dist = np.linalg.norm(top_right_diff, axis=-1)
        nonzero_mask = top_right_dist > 0
        top_right_diff = top_right_diff[nonzero_mask]
        top_right_dist = top_right_dist[nonzero_mask]
        top_right_off = np.mean(top_right_diff[top_right_dist < np.quantile(top_right_dist, quantile_thresh)], axis=0)

        # print(top_right_off)

        # top-left
        top_left_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > 120, pairwise_angle < 150))]
        top_left_dist = np.linalg.norm(top_left_diff, axis=-1)
        nonzero_mask = top_left_dist > 0
        top_left_diff = top_left_diff[nonzero_mask]
        top_left_dist = top_left_dist[nonzero_mask]
        top_left_off = np.mean(top_left_diff[top_left_dist < np.quantile(top_left_dist, quantile_thresh)], axis=0)

        # print(top_left_off)

        # bottom-left
        bottom_left_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle < -120, pairwise_angle > -150))]
        bottom_left_dist = np.linalg.norm(bottom_left_diff, axis=-1)
        nonzero_mask = bottom_left_dist > 0
        bottom_left_diff = bottom_left_diff[nonzero_mask]
        bottom_left_dist = bottom_left_dist[nonzero_mask]
        bottom_left_off = np.mean(bottom_left_diff[bottom_left_dist < np.quantile(bottom_left_dist, quantile_thresh)], axis=0)

        # print(bottom_left_off)

        # bottom-right
        bottom_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle < -30, pairwise_angle > -60))]
        bottom_right_dist = np.linalg.norm(bottom_right_diff, axis=-1)
        nonzero_mask = bottom_right_dist > 0
        bottom_right_diff = bottom_right_diff[nonzero_mask]
        bottom_right_dist = bottom_right_dist[nonzero_mask]
        bottom_right_off = np.mean(bottom_right_diff[bottom_right_dist < np.quantile(bottom_right_dist, quantile_thresh)], axis=0)

        # print(bottom_right_off)

        filled_points = []
        for pt in merged_points:
            filled_points.extend([pt+top_right_off, pt+top_left_off, pt+bottom_left_off, pt+bottom_right_off])
        
        filled_points.extend(merged_points)

        return np.array(filled_points)
    
    merged_points = merge_points(points_grid)
    filled_points = fill_points(merged_points)
    points = np.array(merge_points(filled_points))

    # END

    return points


# def grid2latticeunit(img, proposal, points):
    """
    Convert each lattice grid point into integer lattice grid.

    Hints:
    (1) Since it is difficult to know whether two points should be connected, one way is to map each point into an integer position.
        The integer position should maintain the spatial relationship of these points.
        For instance, if we have three points x1=(50, 50), x2=(70, 50) and x3=(70, 70), we can map them (4, 5), (5, 5) and (5, 6).
        As the distances between (4, 5) and (5, 5), (5, 5) and (5, 6) are both 1, we know that (x1, x2) and (x2, x3) form two edges.
    
    (2) You can use affine transformation to build the mapping above, but do not perform global affine transformation.

    (3) The mapping in the hints above are merely to know whether two points should be connected. 
        If you have your own method for finding the relationship, feel free to implement your owns and ignore the hints above.


    Returns:
        edges: A list of edges in the lattice structure. Each edge is defined by two points. The point coordinate is in the image coordinate.
    """

    # YOUR CODE HERE

    from scipy.spatial.distance import cdist

    def find_avg_offset(points, quantile_thresh=0.01):
        # Find average offset between points

        pairwise_diff = points[np.newaxis, :, :] - points[:, np.newaxis, :]
        pairwise_angle = np.rad2deg(np.arctan2(pairwise_diff[...,0], (pairwise_diff[...,1])))

        # top
        top_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > 60, pairwise_angle < 120))]
        top_right_dist = np.linalg.norm(top_right_diff, axis=-1)
        nonzero_mask = top_right_dist > 0
        top_right_dist = top_right_dist[nonzero_mask]
        top_right = np.mean(top_right_dist[top_right_dist < np.quantile(top_right_dist, quantile_thresh)], axis=0)

        # left
        top_left_diff = pairwise_diff[np.nonzero(np.logical_or(pairwise_angle > 150, pairwise_angle < -150))]
        top_left_dist = np.linalg.norm(top_left_diff, axis=-1)
        nonzero_mask = top_left_dist > 0
        top_left_dist = top_left_dist[nonzero_mask]
        top_left = np.mean(top_left_dist[top_left_dist < np.quantile(top_left_dist, quantile_thresh)], axis=0)

        # bottom
        bottom_left_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > -120, pairwise_angle < -60))]
        bottom_left_dist = np.linalg.norm(bottom_left_diff, axis=-1)
        nonzero_mask = bottom_left_dist > 0
        bottom_left_dist = bottom_left_dist[nonzero_mask]
        bottom_left = np.mean(bottom_left_dist[bottom_left_dist < np.quantile(bottom_left_dist, quantile_thresh)], axis=0)

        # right
        bottom_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle < 30, pairwise_angle > -30))]
        bottom_right_dist = np.linalg.norm(bottom_right_diff, axis=-1)
        nonzero_mask = bottom_right_dist > 0
        bottom_right_dist = bottom_right_dist[nonzero_mask]
        bottom_right = np.mean(bottom_right_dist[bottom_right_dist < np.quantile(bottom_right_dist, quantile_thresh)], axis=0)

        return [top_right, top_left, bottom_left, bottom_right]

    offsets = find_avg_offset(points)
    
    pairwise_dist = cdist(points, points)

    connected_tol = 0.22

    edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            diff = points[i] - points[j]
            angle = np.rad2deg(np.arctan2(diff[0], diff[1]))
            if 45 <= angle < 135:
                 offset = offsets[0]
            elif 135 <= angle or angle < -135:
                offset = offsets[1]
            elif -135 <= angle < 45:
                offset = offsets[2]
            else:
                offset = offsets[3]
            if (1-connected_tol)*offset < pairwise_dist[i,j] < (1+connected_tol)*offset:
                edges.append([points[i].astype(int).tolist(), points[j].astype(int).tolist()])

    # END

    return edges


def grid2latticeunit(img, proposal, points):
    """
    Convert each lattice grid point into integer lattice grid.

    Hints:
    (1) Since it is difficult to know whether two points should be connected, one way is to map each point into an integer position.
        The integer position should maintain the spatial relationship of these points.
        For instance, if we have three points x1=(50, 50), x2=(70, 50) and x3=(70, 70), we can map them (4, 5), (5, 5) and (5, 6).
        As the distances between (4, 5) and (5, 5), (5, 5) and (5, 6) are both 1, we know that (x1, x2) and (x2, x3) form two edges.
    
    (2) You can use affine transformation to build the mapping above, but do not perform global affine transformation.

    (3) The mapping in the hints above are merely to know whether two points should be connected. 
        If you have your own method for finding the relationship, feel free to implement your owns and ignore the hints above.


    Returns:
        edges: A list of edges in the lattice structure. Each edge is defined by two points. The point coordinate is in the image coordinate.
    """

    # YOUR CODE HERE

    from scipy.spatial.distance import cdist

    def find_avg_offset(points, quantile_thresh=0.01):
        # Find average offset between points

        pairwise_diff = points[np.newaxis, :, :] - points[:, np.newaxis, :]
        pairwise_angle = np.rad2deg(np.arctan2(pairwise_diff[...,0], (pairwise_diff[...,1])))

        # top-right
        top_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > 30, pairwise_angle < 60))]
        top_right_dist = np.linalg.norm(top_right_diff, axis=-1)
        nonzero_mask = top_right_dist > 0
        top_right_dist = top_right_dist[nonzero_mask]
        top_right = np.mean(top_right_dist[top_right_dist < np.quantile(top_right_dist, quantile_thresh)], axis=0)

        # top-left
        top_left_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle > 120, pairwise_angle < 150))]
        top_left_dist = np.linalg.norm(top_left_diff, axis=-1)
        nonzero_mask = top_left_dist > 0
        top_left_dist = top_left_dist[nonzero_mask]
        top_left = np.mean(top_left_dist[top_left_dist < np.quantile(top_left_dist, quantile_thresh)], axis=0)

        # bottom-left
        bottom_left_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle < -120, pairwise_angle > -150))]
        bottom_left_dist = np.linalg.norm(bottom_left_diff, axis=-1)
        nonzero_mask = bottom_left_dist > 0
        bottom_left_dist = bottom_left_dist[nonzero_mask]
        bottom_left = np.mean(bottom_left_dist[bottom_left_dist < np.quantile(bottom_left_dist, quantile_thresh)], axis=0)

        # bottom-right
        bottom_right_diff = pairwise_diff[np.nonzero(np.logical_and(pairwise_angle < -30, pairwise_angle > -60))]
        bottom_right_dist = np.linalg.norm(bottom_right_diff, axis=-1)
        nonzero_mask = bottom_right_dist > 0
        bottom_right_dist = bottom_right_dist[nonzero_mask]
        bottom_right = np.mean(bottom_right_dist[bottom_right_dist < np.quantile(bottom_right_dist, quantile_thresh)], axis=0)

        return np.mean([top_right, top_left, bottom_left, bottom_right])

    avg_offset = find_avg_offset(points)
    
    pairwise_dist = cdist(points, points)

    connected_tol = 0.3

    edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if (1-connected_tol)*avg_offset < pairwise_dist[i,j] < (1+connected_tol)*avg_offset:
                edges.append([points[i].astype(int).tolist(), points[j].astype(int).tolist()])

    # END

    return edges
