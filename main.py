from matplotlib import image
from matplotlib import pyplot as plt

import numpy as np

import cv2

from skimage.exposure import match_histograms

import segmentation
import optimal_transport
import graph
import image_reconstruction
import argparse

def harmonize_range(image):
    """
    Assuring that all images have values within [0, 255]
    input:
        - the image of which we want to readjust the range

    returns:
        - the readjusted image
    """

    image = image * 255

    return image

def constitue_three_channels(image):
    """
    Converting a single channel grayscale image into a three channels grayscale image
    inputs:
        - image: a single channel grayscale image

    returns:
        - the three channels grayscale image
    """

    image_final = np.zeros((image.shape[0], image.shape[1], 3))
    image_final[:, :, 0] = image
    image_final[:, :, 1] = image
    image_final[:, :, 2] = image

    return image_final

def synthetization_process(u_path, v_path, num_segments, r, eps, n_iter):
    """
    Compute the whole image synthesis process
    inputs:
        - u: the path of the image of which we want to keep the geometrical features
        - v: the paht image of which we want to keep the color features
        - num_segments: the number of superpixels we want in the images
        - r: the radius of local window
        - eps: the regularizaton parameter
        - n_iter: the number of iterations to process the filtering on

    returns:
        - the synthetized and filtered image
    """

    u = image.imread(u_path)
    v = image.imread(v_path)

    # Readjusting the images so their values are within [0, 255]
    if np.max(u) <= 1.0:
        u = harmonize_range(u)
    if np.max(v) <= 1.0:
        v = harmonize_range(v)
    
    # Constitute three channels images in case of grayscale images
    if len(u.shape) == 2:
        u = constitue_three_channels(u)
    if len(v.shape) == 2:
        v = constitue_three_channels(v)

    segments_u, X, superpixels_u = segmentation.superpixel_segmentation(u, num_segments)
    segments_v, Y, superpixels_v = segmentation.superpixel_segmentation(v, num_segments)

    u = u.astype(int)
    v = v.astype(int)
    superpixels_u = superpixels_u.astype(int)
    superpixels_v = superpixels_v.astype(int)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(u)
    plt.subplot(2, 2, 2)
    plt.imshow(superpixels_u)
    plt.subplot(2, 2, 3)
    plt.imshow(v)
    plt.subplot(2, 2, 4)
    plt.imshow(superpixels_v)

    mu = segmentation.cluster_weights(segments_u)
    nu = segmentation.cluster_weights(segments_v)

    P = optimal_transport.projection_matrix(X, Y, mu, nu)

    superpixels_w = image_reconstruction.superpixels_transfer(P, mu, Y, segments_u)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(superpixels_u)
    plt.subplot(1, 3, 2)
    plt.imshow(superpixels_v)
    plt.subplot(1, 3, 3)
    plt.imshow(superpixels_w)

    w = image_reconstruction.image_synthesis(X, Y, P, mu, segments_u, u)

    w_filtered_final = image_reconstruction.filtered_image(u, w, n_iter, r, eps)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(u)
    plt.subplot(1, 3, 2)
    plt.imshow(w)
    plt.subplot(1, 3, 3)
    plt.imshow(w_filtered_final)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(u)
    plt.subplot(1, 2, 2)
    plt.imshow(w_filtered_final)

    plt.show()

def synthetization_process_constraints(u_path, v_path, num_segments, r, eps, n_iter, n_neighbours, l, rho, k, opt_threshold, tau, opt_iter):
    """
    Compute the whole image synthesis process
    inputs:
        - u: the path of the image of which we want to keep the geometrical features
        - v: the paht image of which we want to keep the color features
        - num_segments: the number of superpixels we want in the images
        - r: the radius of local window
        - eps: the regularizaton parameter
        - n_iter: the number of iterations to process the filtering on
        - n_neighbours: the number of closest neighbours to keep to build the weighted graph of features
        - l: the regularization parameter of divergence of vector field
        - rho: the regularization parameter for tight relaxation
        - k: the dispertion regularization parameter
        - opt_threshold: precision to evaluate convergence
        - tau: updating parameter
        - opt_iter: the number of iterations needed for the gradient descent

    returns:
        - the synthetized and filtered image
    """

    u = image.imread(u_path) # keep only three channels since we do not want to process the alpha channel
    v = image.imread(v_path)
    v = cv2.resize(v, dsize=(u.shape[0], u.shape[1]), interpolation=cv2.INTER_CUBIC)

    # Readjusting the images so their values are within [0, 255]
    if np.max(u) <= 1.5:
        u = harmonize_range(u)
    if np.max(v) <= 1.5:
        v = harmonize_range(v)
    
    # Constitute three channels images in case of grayscale images
    if len(u.shape) == 2:
        u = constitue_three_channels(u)
    if len(v.shape) == 2:
        v = constitue_three_channels(v)
    
    u = match_histograms(u, v, multichannel=True)

    segments_u, X, superpixels_u = segmentation.superpixel_segmentation(u, num_segments)
    segments_v, Y, superpixels_v = segmentation.superpixel_segmentation(v, num_segments)

    u = u.astype(int)
    v = v.astype(int)
    superpixels_u = superpixels_u.astype(int)
    superpixels_v = superpixels_v.astype(int)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(u)
    plt.subplot(2, 2, 2)
    plt.imshow(superpixels_u)
    plt.subplot(2, 2, 3)
    plt.imshow(v)
    plt.subplot(2, 2, 4)
    plt.imshow(superpixels_v)

    mu = segmentation.cluster_weights(segments_u)
    nu = segmentation.cluster_weights(segments_v)

    W = graph.graph_weights(X, mu, segments_u, u)
    #grad_W = graph.graph_grad(W, n_neighbours)

    C = optimal_transport.transport_cost(X, Y)

    #P = optimal_transport.projection_matrix(X, Y, mu, nu)

    P = optimal_transport.relaxation_gurobi(X, Y, W, C, mu, nu, l, rho)
    #P = optimal_transport.relaxation_scipy(X, Y, W, C, mu, nu, l, rho)

    superpixels_w1 = image_reconstruction.superpixels_transfer(P, mu, Y, segments_u)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(superpixels_u)
    plt.subplot(1, 3, 2)
    plt.imshow(superpixels_v)
    plt.subplot(1, 3, 3)
    plt.imshow(superpixels_w1)


    #P = optimal_transport.gradient_descent(X, Y, P, W, C, mu, nu, l, rho, k, opt_iter, opt_threshold, tau)

    superpixels_w = image_reconstruction.superpixels_transfer(P, mu, Y, segments_u)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(superpixels_u)
    plt.subplot(1, 3, 2)
    plt.imshow(superpixels_v)
    plt.subplot(1, 3, 3)
    plt.imshow(superpixels_w)

    w = image_reconstruction.image_synthesis(X, Y, P, mu, segments_u, u)

    w_filtered_final = image_reconstruction.filtered_image(u, w, n_iter, r, eps)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(u)
    plt.subplot(1, 3, 2)
    plt.imshow(w)
    plt.subplot(1, 3, 3)
    plt.imshow(w_filtered_final)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(u)
    plt.subplot(1, 2, 2)
    plt.imshow(w_filtered_final)

    plt.show()

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--image_u", required = True, help = "Path to the image u")
ap.add_argument("-v", "--image_v", required = True, help = "Path to the image v")
ap.add_argument("-s", "--segments", required = False, default=120, type=int, help = "Number of desired segments")
ap.add_argument("-r", "--radius", required = False, default=2, type=int, help = "Size of the radius filtering window")
ap.add_argument("-e", "--epsilon", required = False, default=0.01, type=float, help = "Regularization parameter for post processing")
ap.add_argument("-i", "--iterations", required = False, default=2, type=int, help = "Number of iterations for the post processing filtering")
ap.add_argument("-nn", "--n_neighbours", required = False, default=30, type=int, help = "Number of closest neighbours to keep to build the features graph")
ap.add_argument("-l", "--lambda", required = False, default=0.1, type=float, help = "Regularization parameter of divergence of vector field")
ap.add_argument("-rho", "--rho", required = False, default=1000, type=float, help = "Regularization parameter of tight relaxation")
ap.add_argument("-k", "--k", required = False, default=0.003, type=float, help = "Regularization parameter of dispertion")
ap.add_argument("-ot", "--opt_threshold", required = False, default=0.00001, type=float, help = "Precision threshold used to stop regression")
ap.add_argument("-t", "--tau", required = False, default=0.003, type=float, help = "Regression updating parameter")
ap.add_argument("-oi", "--opt_iter", required = False, default=1000, type=int, help = "Number of maximum iterations for regression")
ap.add_argument("-c", "--constraints", required = False, dest='constraints', action='store_true', help = "Include constraints for colorization or color transfer")
ap.add_argument("-nc", "--no_constraints", required = False, dest='constraints', action='store_false', help = "Do not include constraints for colorization or color transfer")
ap.set_defaults(constraints=True)
args = vars(ap.parse_args())

def main(args):
    """
    Main funcion of the program.
    inputs: 
        - args: the input arguments
    """
    if args["constraints"]:
        synthetization_process_constraints(args["image_u"], args["image_v"], args["segments"], args["radius"], args["epsilon"], args["iterations"], args["n_neighbours"], args["lambda"], args["rho"], args["k"], args["opt_threshold"], args["tau"], args["opt_iter"])
    else:
        synthetization_process(args["image_u"], args["image_v"], args["segments"], args["radius"], args["epsilon"], args["iterations"])

main(args)

    


    


