import numpy as np

import graph
import optimal_transport

def superpixels_transfer(P, mu, Y, segments):
    """
    Attributes new mean color values to superpixels
    inputs:
        - P: the transport matrix of X towards Y
        - mu: the weights of clusters X
        - Y: the clusters of image Y
        - segments: a 2D array which is an integer mask indicating segment labels

    returns:
        - the new superpixel reconstruction
    """

    superpixels = np.zeros((segments.shape[0], segments.shape[1], 3), dtype=int)
    T_U = optimal_transport.color_palette_transfer(P, mu, Y)

    for c in np.unique(segments):
        indexes = np.where(segments==c)
        superpixels[indexes] = np.round(T_U[c]).astype(int)
    
    return superpixels

def image_synthesis(X, Y, P, mu, segments, image_u):
    """
    Synthetizing the new image from the spacial features of image u and using the color palette computed after optimal transfer
    inputs:
        - X: a set of Nx features
        - Y: a set of Ny features
        - P: the transport matrix of X towards Y
        - mu: the weight of cluster X
        - segments: a 2D array which is an integer mask indicating segment labels
        - image_u: the input image
    
    returns:
        - The new image containing the spacial features of image u and the color palette of image v
    """

    T_U = optimal_transport.color_palette_transfer(P, mu, Y) #/ 255
    X[:, 0] = X[:, 0] / image_u.shape[0]*255
    X[:, 1] = X[:, 1] / image_u.shape[1]*255

    image_w = np.zeros((image_u.shape[0] * image_u.shape[1], 3))
    Nx = len(X)

    pixels = np.zeros((image_u.shape[0]*image_u.shape[1], 5))

    i = 0
    for idx, _ in np.ndenumerate(segments):
        pixel = np.array([idx[0]/(segments.shape[0]-1)*255, idx[1]/(segments.shape[1]-1)*255, image_u[idx][0], image_u[idx][1], image_u[idx][2]])
        pixels[i] = pixel
        i += 1
    
    W = np.zeros((image_u.shape[0] * image_u.shape[1],))
    for i in range(Nx):
        indexes = np.where(segments==i)
        V = graph.V_Mahalanobis(indexes, image_u)
        L = (pixels - X[i]) / 255 # likelihood 
        wi = np.exp(-np.sum((L@V)*L, axis=1))
        image_w = image_w + np.outer(wi, T_U[i])
        W += wi

    image_w[:, 0] = np.divide(image_w[:, 0], W)
    image_w[:, 1] = np.divide(image_w[:, 1], W)
    image_w[:, 2] = np.divide(image_w[:, 2], W)
    image_w = image_w.reshape((image_u.shape[0], image_u.shape[1], 3))
    image_w = np.round(image_w).astype(int)
    
    return image_w

def filter(image_u, image_w, r, eps):
    """
    Restore sharp details of the synthetized image using the guided filter
    inputs:
        - image_u: the original image
        - image_w: the synthetized image
        - r: the radius of local window
        - eps: the regularizaton parameter

    returns:
        - The synthetized image with post processing
    """

    h = image_u.shape[0]
    w = image_u.shape[1]
    N = boxfilter(np.ones((h, w)), r)

    mean_u = np.zeros((h, w, 3))
    mean_u[:, :, 0] = boxfilter(image_u[:, :, 0], r) / N
    mean_u[:, :, 1] = boxfilter(image_u[:, :, 1], r) / N
    mean_u[:, :, 2] = boxfilter(image_u[:, :, 2], r) / N

    mean_w = np.zeros((h, w, 3))
    mean_w[:, :, 0] = boxfilter(image_w[:, :, 0], r) / N
    mean_w[:, :, 1] = boxfilter(image_w[:, :, 1], r) / N
    mean_w[:, :, 2] = boxfilter(image_w[:, :, 2], r) / N

    mean_uw = np.zeros((h, w, 3))
    mean_uw[:, :, 0] = boxfilter(image_u[:, :, 0]*image_w[:, :, 0], r) / N
    mean_uw[:, :, 1] = boxfilter(image_u[:, :, 1]*image_w[:, :, 1], r) / N
    mean_uw[:, :, 2] = boxfilter(image_u[:, :, 2]*image_w[:, :, 2], r) / N

    cov_uv = mean_uw - mean_u * mean_w

    mean_u2 = np.zeros((h, w, 3))
    mean_u2[:, :, 0] = boxfilter(image_u[:, :, 0]**2, r) / N
    mean_u2[:, :, 1] = boxfilter(image_u[:, :, 1]**2, r) / N
    mean_u2[:, :, 2] = boxfilter(image_u[:, :, 2]**2, r) / N

    var_u = mean_u2 - mean_u2**2

    a = cov_uv /(var_u + eps)
    b = mean_w - a * mean_u

    mean_a = np.zeros((h, w, 3))
    mean_a[:, :, 0] = boxfilter(a[:, :, 0], r) / N
    mean_a[:, :, 1] = boxfilter(a[:, :, 1], r) / N
    mean_a[:, :, 2] = boxfilter(a[:, :, 2], r) / N

    mean_b = np.zeros((h, w, 3))
    mean_b[:, :, 0] = boxfilter(b[:, :, 0], r) / N
    mean_b[:, :, 1] = boxfilter(b[:, :, 1], r) / N
    mean_b[:, :, 2] = boxfilter(b[:, :, 2], r) / N

    q = mean_a * image_u + mean_b

    return q

def boxfilter(image, r):
    """
    Box filtering using cumulative sum
    inputs:
        - image: The image to be filtered
        - r: the radius of local window
    
    returns: 
        - Filtered image
    """

    h = image.shape[0]
    w = image.shape[1]
    imDst = np.zeros((h, w))

    # Cumulative sum over Y axis
    imCum = np.cumsum(image, axis=0)

    # Difference over Y axis
    imDst[:r+1, :] = imCum[r:2*r+1, :]
    imDst[r+1:h-r, :] = imCum[2*r+1:h, :] - imCum[:h-2*r-1, :]
    imDst[h-r:h, :] = np.tile(imCum[h-1, :], [r, 1]) - imCum[h-2*r-1:h-r-1, :]

    # Cumulative sum over X axis 
    imCum = np.cumsum(imDst, axis=1)

    # Difference over X axis 
    imDst[:, :r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:w-r] = imCum[:, 2*r+1:w] - imCum[:, :w-2*r-1]
    imDst[:, w-r:w] = np.tile(imCum[:, w-1].reshape((imCum.shape[0], 1)),[1, r]) - imCum[:, w-2*r-1:w-r-1]

    return imDst

def filtered_image(image_u, image_w, n_iter, r, eps):
    """
    Apply the filter to the synthetized image to restore the sharp details lost during the optimal transport process
    inputs:
        - image_u: the original image
        - image_w: the synthetized image after optimal transport
        - n_iter: the number of iterations to process the filtering on
        - r: the radius of local window of the NLMR filter
        - eps: the regularizaton parameter of the NLMR filter

    returns: 
        - the filtered synthetized image
    """
    image_u = image_u / 255
    w_temp = image_w / 255 - image_u
    for i in range(n_iter):
        w_filtered = filter(image_u, w_temp, r, eps)
        w_temp = w_filtered

    w_filtered_final = np.round((image_u + w_temp)*255).astype(int)

    indexes_min = np.where(w_filtered_final < 0)
    indexes_max = np.where(w_filtered_final > 255)

    while len(indexes_min[0]) > 0:
        w_filtered_final[indexes_min] = 0 # in case of NaN
        for id in range(len(indexes_min[0])):
            l_bound_ax0 = indexes_min[0][id] - 1
            l_bound_ax1 = indexes_min[1][id] - 1
            u_bound_ax0 = indexes_min[0][id] + 2
            u_bound_ax1 = indexes_min[1][id] + 2
            if l_bound_ax0  < 0:
                l_bound_ax0 = 0
            if l_bound_ax1 < 0:
                l_bound_ax1 = 0
            if u_bound_ax0 > image_u.shape[0]:
                u_bound_ax0 = - 1
            if u_bound_ax1 >  image_u.shape[1]:
                u_bound_ax1 = - 1
            interpolation_raw = w_filtered_final[l_bound_ax0:u_bound_ax0, l_bound_ax1:u_bound_ax1, indexes_min[2][id]].copy()
            positive_interpolation_idx = np.where(interpolation_raw > 0)
            interpolation = interpolation_raw[positive_interpolation_idx]
            if len(interpolation) >= 1:
                w_filtered_final[indexes_min[0][id], indexes_min[1][id], indexes_min[2][id]] = np.median(interpolation)
        indexes_min = np.where(w_filtered_final < 0)
    
    while len(indexes_max[0]) > 0:
        w_filtered_final[indexes_max] = 0 # in case of NaN
        for id in range(len(indexes_max[0])):  
            l_bound_ax0 = indexes_max[0][id] - 1
            l_bound_ax1 = indexes_max[1][id] - 1
            u_bound_ax0 = indexes_max[0][id] + 2
            u_bound_ax1 = indexes_max[1][id] + 2
            if l_bound_ax0 == -1:
                l_bound_ax0 = 0
            if l_bound_ax1 == -1:
                l_bound_ax1 = 0
            if u_bound_ax0 == image_u.shape[0] + 1:
                u_bound_ax0 = - 1
            if u_bound_ax1 == image_u.shape[1] + 1:
                u_bound_ax1 = - 1
            interpolation_raw = w_filtered_final[l_bound_ax0:u_bound_ax0, l_bound_ax1:u_bound_ax1, indexes_max[2][id]].copy()
            positive_interpolation_idx = np.where(interpolation_raw > 0)
            interpolation = interpolation_raw[positive_interpolation_idx]
            if len(interpolation) >= 1:
                w_filtered_final[indexes_max[0][id], indexes_max[1][id], indexes_max[2][id]] = np.median(interpolation)
        indexes_max = np.where(w_filtered_final > 255)

    return w_filtered_final