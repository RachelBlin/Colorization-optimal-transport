import numpy as np
import ot

def transport_cost(X, Y):
    """
    Compute the matrix CXY in which CXY(i,j) is the cost to transport X(i) towards Y(j)
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
    
    returns:
        - the transport cost of X towards Y, a matrix of shape Nx*Ny
    """

    Nx = len(X)
    Ny = len(Y)
    CXY = np.zeros((Nx, Ny))

    for idx, _ in np.ndenumerate(CXY):
        CXY[idx] = np.linalg.norm(X[idx[0], 2:]-Y[idx[1], 2:])**2 # Only select the color features 
    
    return CXY

def transport_map(X, Y, mu, nu):
    """
    Compute the transport map from image u to image v
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
        - mu: the weights of features (superpixels) of image u
        - nu: the weights of features (superpixels) of image v

    returns:
        - the projection matrix from image u to image v
    """

    C = transport_cost(X, Y) # cost matrix between features X and features Y
    P = ot.emd(mu, nu, C) # Basic solver

    return P