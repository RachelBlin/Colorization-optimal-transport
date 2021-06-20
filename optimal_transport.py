import numpy as np
import ot

import graph

import gurobipy as gp
from gurobipy import GRB

from numpy import matlib as mb

from scipy.optimize import minimize, Bounds

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

    X_norm = X.copy()

    X_norm[:, 0] = X[:, 0] / 500
    X_norm[:, 1] = X[:, 1] / 500
    X_norm[:, 2:] = X[:, 2:] / 255

    Y_norm = Y.copy()

    Y_norm[:, 0] = Y[:, 0] / 500
    Y_norm[:, 1] = Y[:, 1] / 500
    Y_norm[:, 2:] = Y[:, 2:] / 255

    for idx, _ in np.ndenumerate(CXY):
        #CXY[idx] = np.linalg.norm((X[idx[0], 2:]-Y[idx[1], 2:]) / 255)**2 # Only select the color features 
        #CXY[idx] = np.linalg.norm((X[idx[0], :2]-Y[idx[1], :2]))**2 # Only select the spacial features 
        CXY[idx] = np.linalg.norm((X_norm[idx[0]]-Y_norm[idx[1]]))**2 # Select both the spacial and color features 
        #CXY[idx] = 0.5*np.linalg.norm((X_norm[idx[0], :2]-Y_norm[idx[1], :2]))**2 + 0.5*np.linalg.norm((X_norm[idx[0], 2:]-Y_norm[idx[1], 2:]) / 255)**2
    
    return CXY

def projection_matrix(X, Y, mu, nu):
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

def color_palette_transfer(P, mu, Y):
    """
    A function to transfer the color features of clusters Y to clusters X knowing the transport matrix.
    inputs:
        - P: the transport matrix of X towards Y
        - mu: the weights of clusters X
        - Y: the clusters of image Y

    returns:
        - the color features after transfer 
    """

    T_U = np.diag(1/mu)@P@Y[:, 2:] # select only color features of Y

    return T_U

def regularization_constraints(X, Y, P, W, C, mu, nu, l, rho, k):
    """
    Compute the objective function including regularization constraints of the optimal transport problem
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
        - P: the projection matrix from image u to image v
        - W: the weighted graph between features X
        - C: the transport cost of X towards Y, a matrix of shape Nx*Ny
        - mu: the weights of features (superpixels) of image u
        - nu: the weights of features (superpixels) of image v
        - l: the regularization parameter of divergence of vector field
        - rho: the regularization parameter for tight relaxation
        - k: the dispertion regularization parameter

    returns:
        - The function we want to relax
    """

    T_U = color_palette_transfer(P, mu, Y) # Basic transport map from X to Y without constraints
    V = T_U - X[:, 2:] # regularization of average transport displacement
    div = divergence_vector_field(V, W) # divergence of vector field
    J = np.linalg.norm(div, ord=1) # regularity of the flow
    #J = 0.5 * np.linalg.norm(np.dot(np.diag(mu), np.reshape(W @ V, (X.shape[0], X.shape[0], 3)))) ** 2
    I = np.linalg.norm(np.ones((Y.shape[0], )) - (P.T @ np.ones((X.shape[0], ))/nu), ord=1) # check that the relaxation is tight
    #I = 0.5 * np.linalg.norm(np.dot(np.ones(X.shape[0]), P @ np.sqrt(np.diag(nu)))) ** 2 - 0.5
    D = np.sum(P * (np.ones(Y.shape[0]) @ np.diag(Y[:, 2:] @ Y[:, 2:].T).T - np.diag(mu) @ P @ Y @ Y.T))

    func = np.sum(C*P) + l * J + rho * I + k * D

    return func

def regularization_constraints_grad(X, Y, P, W, C, mu, nu, l, rho, k):
    """
    Compute gradiant of objective function with the regularization constraints 
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
        - P: the projection matrix from image u to image v
        - W: the weighted graph between features X
        - C: the transport cost of X towards Y, a matrix of shape Nx*Ny
        - mu: the weights of features (superpixels) of image u
        - nu: the weights of features (superpixels) of image v
        - l: the regularization parameter of divergence of vector field
        - rho: the regularization parameter for tight relaxation
        - k: the dispertion regularization parameter

    returns: 
        - the gradient of the projection matrix
    """ 

    T_U = color_palette_transfer(P, mu, Y)
    V = (T_U - X[:, 2:]) / 255

    scaling = np.repeat(mu ** 2, X.shape[0])[:, np.newaxis]
    #grad_J = np.diag(1/mu) @ (W.T @ (scaling * (W @V))) @ (Y[:, 2:].T / 255)
    grad_J = np.diag(1/mu) @ (W.T @ (W @ V)) @ (Y[:, 2:].T / 255)
    grad_I = np.ones((X.shape[0], X.shape[0])) @ P @ np.diag(1/nu)
    grad_D = np.ones(Y.shape[0]) @ np.diag(Y @ Y.T).T - 2 * np.diag(mu) @ P @ Y @ Y.T

    grad = C + l * grad_J + rho * grad_I + k * grad_D

    return grad

def relaxation_gurobi(X, Y, W, C, mu, nu, l, rho):
    """
    Compute the objective function including regularization constraints of the optimal transport problem
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
        - P: the projection matrix from image u to image v
        - W: the weighted graph between features X
        - C: the transport cost of X towards Y, a matrix of shape Nx*Ny
        - mu: the weights of features (superpixels) of image u
        - nu: the weights of features (superpixels) of image v
        - l: the regularization parameter of divergence of vector field
        - rho: the regularization parameter for tight relaxation

    returns:
        - The function we want to relax
    """

    Nx = X.shape[0]
    Ny = Y.shape[0]
    C = (C.T).reshape((Nx*Ny,))
    mu_r = np.diag(mb.repmat(1/mu, 1, Ny)[0])

    Y_mul = np.zeros((3*Nx, Ny*Nx))

    X_mul = np.zeros((3*Nx,))

    for i in range(3):
        X_mul[i*Nx:(i+1)*Nx] = X[:, i+2]

    for i in range(3):
        for j in range(Nx):
            Y_mul[i*Nx+j, j::Nx] = Y[:, i+2]

    # Create a new model
    m = gp.Model("matrix2")

    # Create variables
    P = m.addMVar(Nx*Ny, lb=np.zeros((Nx*Ny,)), ub=np.ones((Nx*Ny,)), name="P")
    k = m.addMVar(Ny, lb=0, name="k")

    D = Y_mul @ mu_r

    # Set objective
    #m.setObjective(C @ P + l * np.linalg.norm(divergence_vector_field(color_palette_transfer(P, mu, Y) - X[:, 2:], W), ord=1) + rho * np.linalg.norm(k - np.ones((Y.shape[0], ))), GRB.MINIMIZE)
    m.setObjective(C @ P, GRB.MINIMIZE) #+ l * D @ P - l * X_mul, GRB.MINIMIZE)

    A = np.zeros((Nx, Nx*Ny))
    B = np.zeros((Ny, Nx*Ny))

    for i in range(Nx):
        A[i, i:Nx*Ny:Nx] = 1

    for j in range(Ny):
        B[j, j*Nx:(j+1)*Nx] = 1

    # Add constraints:
    #b = P.T @ np.ones((X.shape[0],))
    #m.addConstr(k >= 0, "c0")
    #m.addConstr(P >= 0, "c1")
    #m.addConstr(P <= 1, "c2")
    m.addConstr(A @ P == mu, "c3")
    m.addConstr(B @ P == nu, "c4")
    #m.addConstr(P @ B <= k * nu, "c4")
    #m.addConstr(np.sum(k * nu) >= 1, "c5")

    # Optimize model
    m.optimize()

    solution = P.X
    
    solution = solution.reshape((Ny, Nx)).T
    print(np.max(solution), np.min(solution))

    return solution #, k

def relaxation_scipy(X, Y, W, C, mu, nu, l, rho):
    """
    Compute the objective function including regularization constraints of the optimal transport problem
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
        - P: the projection matrix from image u to image v
        - W: the weighted graph between features X
        - C: the transport cost of X towards Y, a matrix of shape Nx*Ny
        - mu: the weights of features (superpixels) of image u
        - nu: the weights of features (superpixels) of image v
        - l: the regularization parameter of divergence of vector field
        - rho: the regularization parameter for tight relaxation

    returns:
        - The function we want to relax
    """

    Nx = X.shape[0]
    Ny = Y.shape[0]
    fun = lambda P: np.sum(C*P)
    x0 = np.ones((Nx, Ny))

    cons = ({'type': 'eq', 'fun': lambda P:  P.T @ np.ones((Nx, )) == nu},
            {'type': 'eq', 'fun': lambda P:  P @ np.ones((Ny, )) == mu})
    
    bnds = (np.zeros((Nx, Ny)), np.ones((Nx, Ny)))

    res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)

    return res

def superpixels_corrdinatelikelihood(X, Y, W, C, mu, nu, l, rho):
    """
    Compute the objective function including regularization constraints of the optimal transport problem
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
        - P: the projection matrix from image u to image v
        - W: the weighted graph between features X
        - C: the transport cost of X towards Y, a matrix of shape Nx*Ny
        - mu: the weights of features (superpixels) of image u
        - nu: the weights of features (superpixels) of image v
        - l: the regularization parameter of divergence of vector field
        - rho: the regularization parameter for tight relaxation

    returns:
        - The function we want to relax
    """

def gradient_descent(X, Y, P, W, C, mu, nu, l, rho, k, iter, eps, tau):
    """
    Perform gradient descent to apply contraints on the projection matrix
    inputs:
        - X: the first set of features of size Nx of which we want to keep the spatial features
        - Y: the second set of features of size Ny of which we want to keep the color features
        - P: the projection matrix from image u to image v
        - W: the weighted graph between features X
        - C: the transport cost of X towards Y, a matrix of shape Nx*Ny
        - mu: the weights of features (superpixels) of image u
        - nu: the weights of features (superpixels) of image v
        - l: the regularization parameter of divergence of vector field
        - rho: the regularization parameter for tight relaxation
        - k: the dispertion regularization parameter
        - iter: the number of iterations needed for the gradient descent
        - eps: precision to evaluate convergence
        - tau: updating parameter

    returns:
        - the regularized projection matrix
    """

    func = regularization_constraints(X, Y, P, W, C, mu, nu, l, rho, k)
    func_previous = 2 * func

    i = 0
    while i < iter and np.abs(func - func_previous)/func_previous > eps:
        print(i)
        func_previous = func
        grad = regularization_constraints_grad(X, Y, P, W, C, mu, nu, l, rho, k)
        P = P - tau*grad
        P = simplex_matrix_projection(P, mu)
        func = regularization_constraints(X, Y, P, W, C, mu, nu, l, rho, k)
        i += 1
    
    return P

def divergence_vector_field(V, W):
    """
    Computes the divergence of a vector field V according to feature X
    inputs:
        - V: the vector field on which we compute the divergence
        - W: the weighted graph between features X
    """

    Nx = W.shape[0]
    div = np.zeros((Nx, 3))

    for j in range(Nx):
        div[j, 0] = np.sum(W[j, :]*(-V[:, 0] + V[j, 0]))
        div[j, 1] = np.sum(W[j, :]*(-V[:, 1] + V[j, 1]))
        div[j, 2] = np.sum(W[j, :]*(-V[:, 2] + V[j, 2]))
    
    return div

def pseudo_simplex_projection(Pi, mui):
    """
    Project a vector Pi on the set of positive vectors whose coefficients sum to mui. 
    inputs:
        - Pi: line i of projection matrix P
        - mui: weight of cluster Xi
    
    returns:
        - The projected vector
    """

    Ny = len(Pi)
    sort_Pi = np.sort(Pi)

    i = Ny - 2
    t_i = (np.sum(sort_Pi[i + 1:Ny]) - mui)  / (Ny - i - 1)

    while t_i < sort_Pi[i] and i >= 0:
        i = i - 1
        t_i = (np.sum(sort_Pi[i + 1:Ny]) - mui) / (Ny - i - 1)

    if i < 0:
        t_hat = (np.sum(Pi) - mui) / Ny
    else:
        t_hat = t_i

    Pi_proj = np.maximum(Pi - t_hat, 0)

    return Pi_proj

def simplex_matrix_projection(P, mu):
    """
    Projecting matrix P so that its rows sum to wheight of clusters X
    inputs:
        - P: the matrix to be projected
        - mu: the weights of clusters X

    returns:
        - The projection of matrix P
    """

    P_proj = P.copy()

    for i in range(P.shape[0]):
        P_proj[i] = pseudo_simplex_projection(P[i], mu[i])
    
    return P_proj