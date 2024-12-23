import numpy as np
import random, copy, cv2

def get_bbox_binary(seg, jitter=0.):
    """
    This function calculates a bounding box for a binary image.
    If jitter is 0, then the bbox will be close aroung GT, if it is > 0, then the bbox will be larger.
    """
    bounding_boxes, bbox = [], []

    seg_idxs = np.argwhere(seg == 1)

    if np.any(seg_idxs):    # Only True if array is not empty
        X1 = np.int32(np.min(seg_idxs[:,1]))
        X2 = np.int32(np.max(seg_idxs[:,1]))
        Y1 = np.int32(np.min(seg_idxs[:,0]))
        Y2 = np.int32(np.max(seg_idxs[:,0]))

        # Add jittering, i.e. expand bbox based on jitter value
        X1, X2, Y1, Y2 = np.int32(X1*(1-jitter)), np.int32(X2*(1+jitter)), np.int32(Y1*(1-jitter)), np.int32(Y2*(1+jitter))
            
        bbox = [X1, X2, Y1, Y2]   # Points for cv plotting: (X1, Y1) -- (X2, Y2) being (Xmin, Xmax) -- (Ymin, Ymax), i.e. X1 or Y1 always > X2 or Y2
    
    if len(bbox) == 0 or not np.any(seg_idxs):
        bbox = []

    bounding_boxes.append(bbox)
        
    return bounding_boxes

def get_bbox_per_cc_binary(seg, jitter=0.):
    """
    This function calculates a bounding box for a binary image by focusing on all CCs.
    If jitter is 0, then the bbox will be close aroung GT, if it is > 0, then the bbox will be larger:
    https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
    """
    bounding_boxes, bbox = [], []
    # NOTE: This extracts for every CC a bounding box, however if this is used, no samples can be used for SAM!
    # Get nr of CCs
    num_labels, labels, _, _ = get_ccs_for_seg(seg)

    # Loop through CCs and get bbox for every CC
    for i in range(1, num_labels):
        seg_idxs = np.argwhere(labels == i)

        if np.any(seg_idxs):    # Only True if array is not empty
            X1 = np.int32(np.min(seg_idxs[:,1]))
            X2 = np.int32(np.max(seg_idxs[:,1]))
            Y1 = np.int32(np.min(seg_idxs[:,0]))
            Y2 = np.int32(np.max(seg_idxs[:,0]))

            # Add jittering, i.e. expand bbox based on jitter value
            X1, X2, Y1, Y2 = np.int32(X1*(1-jitter)), np.int32(X2*(1+jitter)), np.int32(Y1*(1-jitter)), np.int32(Y2*(1+jitter))
                
            bbox = [X1, X2, Y1, Y2]   # Points for cv plotting: (X1, Y1) -- (X2, Y2) being (Xmin, Xmax) -- (Ymin, Ymax), i.e. X1 or Y1 always > X2 or Y2

            bounding_boxes.append(bbox)
        
    return bounding_boxes

def get_center_point_from_bbox(bboxs):
    """
    This function takes a list of 4 coordinates representing a bbox and returns the center point of this bbox.
    bounding_boxes = [X1, X2, Y1, Y2].
    """
    center_points = []
    for box in bboxs:
        if len(box) == 0:  # <-- Empty, i.e. there is no bbox
            center_points.append([])
        else:
            [X1, X2, Y1, Y2] = copy.deepcopy(box)   # <-- Otherwise in-place changes
            center_point = (np.int32(((Y1+Y2)/2)), np.int32((X1+X2)/2))
            center_points.append(center_point)
    return center_points

def get_centroid_of_gt_mask_per_CC(slice):
    """
    This function gets a segmentation GT mask and returns the centroid point of this mask for every CC:
    https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
    """
    centroids_ = []
    num_labels, _, _, centroids = get_ccs_for_seg(slice)
    
    for i in range(1, num_labels):
        #centroids_.append(centroids[i])
        centroids_.append(centroids[i][::-1])
    return centroids_

def get_random_point_per_quadrant(slice):
    """
    This function gets a slice and all bboxes for every CC. It then splits every CC in 4 parts and extract one random
    sample from each quadrant: https://stackoverflow.com/questions/60325518/make-everything-zeros-in-a-numpy-2d-array-outside-of-bounding-box
    """
    samples = []
    slice = slice.squeeze()
    #bboxs = get_bbox_binary(slice)  # uses one large bbox and splits it --> 4 points in total
    bboxs = get_bbox_per_cc_binary(slice)   # Uses n bboxs based on nr of CCs --> nr of CCs * 4 points if possible
    if isListEmpty(bboxs):
        return samples
    center_pts = get_center_point_from_bbox(bboxs)
    for (Y, X), box in zip(center_pts, bboxs):  # (X, Y) is the center point of the bbox where all 4 boxes meet
        [X1, X2, Y1, Y2] = copy.deepcopy(box) # (X1, Y1) --> top left corner -- (X2, Y2) bottom right corner
        # Split box into 4 quadrants        
        b_top_left = [X1, X, Y1, Y] # (X1, Y1) --> top left corner -- (X, Y) bottom right corner (now the center of big bbox)
        b_top_right = [X, X2, Y1, Y]  # (X, Y1) --> top left corner -- (X2, Y) bottom right corner
        b_bottom_left = [X1, X, Y, Y2] # (X1, Y) --> top left corner -- (X, Y2) bottom right corner
        b_bottom_right = [X, X2, Y, Y2] # (X, Y) top left corner (now the center of big bbox) -- (X2, Y2) --> bottom right corner
        q_boxs = [b_top_left, b_top_right, b_bottom_left, b_bottom_right]
        for q_box in q_boxs:
            # Get the points within the area of q_box from the segmentation mask
            res = np.zeros_like(slice)
            [X1, X2, Y1, Y2] = copy.deepcopy(q_box)
            res[Y1:Y2+1, X1:X2+1] = slice[Y1:Y2+1, X1:X2+1]
            seg_idxs = np.argwhere(res == 1)
            if not isListEmpty(seg_idxs):   # <-- Sometimes bbox is soo large or soo small that the quarter bbox does not contain any labels
                # Get one random sample from those ids
                myrandom = random.Random(42)
                samples.append(list(myrandom.choice(seg_idxs)))
    #print(samples)
    return samples

def get_ccs_for_seg(seg):
    r"""
    This function gets a segmentation mask (slice) and returns the CCs with stats using opencv.
    NOTE: ID=0 represents the background.
    """
    connectivity = 4    # 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg.squeeze(), connectivity, cv2.CV_32S)
    return num_labels, labels, stats, centroids

def isListEmpty(inList):
    """
    Checks if a list is empty (can also be a nested list):
    https://stackoverflow.com/questions/11295609/how-can-i-check-whether-a-numpy-array-is-empty-or-not
    """
    return np.asarray(inList).size == 0

def load_npzfile(path, nr_samples, jitter=0., use_only_centroid_of_gt=False, use_only_center_of_bbox=False, use_quarter_four_points=False, **kwargs):
    """
    This function loads a single npz file and returns the values.
    """
    vol = np.load(path)
    orig_sizes = [vol['original_size']]
    input_sizes = [vol['input_size']]
    neg_samples_gt, samples_gt, bboxs_gt = list(), list(), list()

    imgs = [vol['imgs']]
    imgs = np.concatenate(imgs, axis=0)
    segs = [vol['gts'][:, np.newaxis, ...]]
    # find the contours
    for seg in segs:
        for slice in seg:
            bboxs = get_bbox_binary(slice.squeeze(), jitter=jitter) # compute the bounding rectangle of the contour w/o using CCs
            bboxs_gt.append(bboxs)

            if use_only_centroid_of_gt:
                samples_gt.append(get_centroid_of_gt_mask_per_CC(slice))
            elif use_only_center_of_bbox:
                samples_gt.append(get_center_point_from_bbox(bboxs))
            elif use_quarter_four_points:
                samples_gt.append(get_random_point_per_quadrant(slice))
            else:   # If those are not set, then extract as usual
                
                if isListEmpty(bboxs):  # <-- Empty, i.e. there are no bboxs/CCs
                    samples_gt.append([])
                else: # This only selects n random samples from the GT masks were labels are != 0
                    seg_idxs = np.argwhere(slice == 1)
                    # Set a specific random object here so every image get always the same random n points
                    myrandom = random.Random(42)
                    # nr_samples_ = nr_samples if len(seg_idxs) > nr_samples else len(seg_idxs)   # Make sure we don't get more number of samples than idxs
                    samples_gt.append([[x[1], x[2]] for x in myrandom.choices(seg_idxs, k=nr_samples)])

                    # TODO: Do the sanity check here after they have been sampled
                    # TODO: Niklas: Make sure that we sample at least one point from ever CC (nr CCs = len(bboxs))
            
            # -- Always extract negative points, they are only used if the flag is set during training -- #
            seg_idxs = np.argwhere(slice != 1)
            if len(seg_idxs) > 0:    # <-- it might be that a slice is fully segmented, e.g. for the Task900 we use (BCSS)
                # Set a specific random object here so every image get always the same random n points
                myrandom = random.Random(42)
                nr_samples_ = nr_samples if len(seg_idxs) > nr_samples else len(seg_idxs)   # Make sure we don't get more number of samples than idxs
                neg_samples_gt.append([[x[1], x[2]] for x in myrandom.choices(seg_idxs, k=nr_samples_)])
            else:
                neg_samples_gt.append([])   # <-- So samples_gt and neg_samples_gt have the same amount of lists

    segs = np.concatenate(segs, axis=0)
    embeds = [vol['img_embeddings']]
    embeds = np.concatenate(embeds, axis=0)

    # transformed image (b: (n, 256, 256, 3)), segmentation masks (b: (n, 1, 256, 256)), embeddings (b: (n, 1, 256, 64, 64)), bboxs_gt (b: (n, 4)), samples_gt (b: (n, nr_samples, 2))
    return (imgs, segs, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt, neg_samples_gt)


def load_niifile(
    filename
):
    """
    Loads a file in nii.gz format which is NOT pre-processed using one of our scripts.

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
    """
    # TODO: In case we did not pre-process but just have the simple nifti image for inference
    # --> or make an inference pre-procesing step here or even script?
    pass