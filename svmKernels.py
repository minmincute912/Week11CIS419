"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    global _polyDegree

    dot_product = np.dot(X1, X2.T)
    polynomialKernelMatrix = (dot_product + 1)** _polyDegree
    return polynomialKernelMatrix



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''

    global _gaussSigma

    pairwise_distance = np.sum(X1**2 , axis = 1).reshape(-1,1) + np.sum(X2**2, axis = 1) - 2*np.dot(X1,X2.T)
    gaussian_kernel_matrix = np.exp(-pairwise_distance/ ( 2 * _gaussSigma ** 2))

    return gaussian_kernel_matrix


    return #TODO



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return #TODO (CIS 519 ONLY)

