import numpy as np
from scipy.sparse import csr_matrix

def graph_weights(X, mu, segments, image):
    """
    Compute the weights between all the clusters of X pairwise to define the neighborhood between them
    inputs: 
        - X: a set of Nx features
        - segments: a 2D array which is an integer mask indicating segment labels
        - image: the input image
        - n_neighbours: number of closest neighbours to keep
    returns:
        - W, a symmetric matrix in which W(i,j) indicates the wheight of the edge that links X(i) to X(j)
    """

    Nx = len(X)
    W = np.zeros((Nx, Nx))

    X_norm = X.copy()
    X_norm[:, 0] = X[:, 0] / (segments.shape[0]-1) #* 255
    X_norm[:, 1] = X[:, 1] / (segments.shape[1]-1) #* 255
    X_norm[:, 2:] = X[:, 2:] / 255

    for idx, _ in np.ndenumerate(W):
        if W[idx[1], idx[0]] > 0:
            W[idx] = W[idx[1], idx[0]] # W symmetric
        if idx[1] != idx[0]: # we do not want the vertice connected with itself
            indexes = np.where(segments==idx[0])
            V = V_Mahalanobis(indexes, image)
            M_dist = mu[idx[0]] * Mahalanobis_distance(X_norm[idx[0]] - X_norm[idx[1]], V)
            #M_dist = 1/ np.linalg.norm(X_norm[idx[0]] - X_norm[idx[1]])
            W[idx] = np.exp(-0.5*M_dist)

    return W

def graph_grad(W, n_neighbours):
    """
    Build the gradient of the weighted graph
    """

    Nx = W.shape[0]

    grad_W = np.zeros((Nx, Nx, Nx))
    for i in range(Nx):
        best_vertices = np.argsort(-W[i, :])[:n_neighbours]
        grad_W[i, best_vertices, i] = W[i, best_vertices]
        grad_W[i, best_vertices, best_vertices] = - W[i, best_vertices]
    
    grad_W = np.reshape(grad_W, (Nx ** 2, Nx))

    grad_W = csr_matrix(grad_W)

    return grad_W

def Mahalanobis_distance(X, Z):
    """
    Comput the Mahalanobis distance : ||X||²_Z = X'*Z^(-1)*X
    inputs:
        - X: the element of which we want to comput the Mahalanobis distance
        - Z: the covariance matrix used to compute the Mahalanobis distance

    returns:
        - The Mahalanobis distance
    """

    Z_inv = pseudo_inverse(Z)

    M_dist = X.T@Z_inv@X

    return M_dist

def pseudo_inverse(Z):
    """
    Compute the speudo-inverse of a matrix Z
    inputs:
        - Z: the matrix of which we want to compute the pseudo inverse
    
    returns:
        - The pseudo inverse of Z
    """

    Z_inv = np.linalg.pinv(Z)

    return Z_inv

def V_Mahalanobis(indexes, image):
    """
    Compute matrix V = 1/mu *sum_E,F∈Ni(S*(E-X)*(F-X)'*S') used to compupte the Mahalanobis distance
    inputs:
        - indexes: a tuple of numpy arrays referring to the coordinates of the pixels which belong to cluster X
        - image: the input image
    
    returns:
        - matrix V 
    """

    V_filter = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]]) # dissociate spatial and color information
    #V_filter = np.ones((5, 5)) # do  not dissociat spacial and color information

    #var = np.array([1, 1, 1, 1, 1])
    var = np.array([1, 1, 0.1, 0.1, 0.1])

    scal = np.ones((5, 5))
    #scal = 2 * np.ones((5, 5)) - np.eye(5)

    V_filter = (var.T@var)*V_filter

    eps = 0.0001 * np.eye(5) 

    data_cov = np.zeros((len(indexes[0]), 5), dtype=int)
    data_cov[:, 0] = indexes[0]/(image.shape[0]-1) * 255
    data_cov[:, 1] = indexes[1]/(image.shape[1]-1) * 255
    data_cov[:, 2:] = image[indexes] 

    cov = np.cov(data_cov, rowvar=False)

    V = np.linalg.inv(cov/(255**2)*V_filter + eps)*scal

    return V

