import numpy as np

from skimage.segmentation import slic  # super pixels
from skimage import color

def superpixel_segmentation(image, num_segments):
    """
    Segment an image into an approximative number of superpixels.
    inputs:
        -image: a numpy array to be segmented
        -num_segments: the number of of desired segments in the segmentation

    returns:
        - the number of segments after superpixel segmentation
        - the content of each cluster (center coordinates, mean of color)
    """
    segments = slic(image, n_segments=num_segments, sigma=5) # segmentation using the k-means clustering
    clusters = np.zeros((len(np.unique(segments)), 5))
    superpixels = color.label2rgb(segments, image, kind='avg')

    for c in np.unique(segments):
        indexes = np.where(segments==c)
        mean_color = superpixels[indexes[0][0], indexes[1][0]] # just access the first element of superpixel since all elements of superpixel are average color of cluster
        mean_indexes = np.array([int(np.round(np.mean(indexes[0]))), int(np.round(np.mean(indexes[1])))])
        clusters[c] = np.concatenate((mean_indexes, mean_color), axis=None)

    return segments, clusters, superpixels

def cluster_weights(segments):
    """
    Compute the weights of each cluster.
    inputs:
        - segments: a 2D array which is an integer mask indicating segment labels

    returns: 
        - a list of the weights associated to each cluster given the following formula N_pixel_cluster/N_pixel_image
    """

    size_image = segments.shape[0]*segments.shape[1]
    weights = np.zeros((len(np.unique(segments)),))

    for c in np.unique(segments):
        indexes = np.where(segments==c)
        n_pixels = len(indexes[0])
        weights[c] = n_pixels/size_image
    
    return weights
     