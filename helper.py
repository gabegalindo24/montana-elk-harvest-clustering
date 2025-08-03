# Gabe Galindo
# CYBR 304 - Foundation of Computational Mathematics
# Final Project - K-Means Clustering
# 05/15/2025

import numpy as np

def main():
    '''
    Main function to use for personal testing.
    Nothing in this function will affect the autotests.
    '''
    pass


def multiply(A, B):
    '''
    Function that takes two compatible matrices and multiplies them together.
    param A: matrix in dimensions m by n
    param B: matrix in dimensions n by m
    return: product of the matrices
    '''
    if len(A[0]) != len(B): # check for ensuring multiplication of two compatible matrices
        return "ERROR, the two matrices are non-compatible!"

    product = []
    for row in range(len(A)): # loop through matrix A rows
        product.append([])
        for col in range(len(B[0])): # loop through matrix B columns
            accum = 0
            for k in range(len(A[0])): # loop through matrix A columns
                 accum += A[row][k] * B[k][col] # multiply correct elements and add to accum
            product[row].append(accum)

    return np.array(product) # return product matrix


def transpose(A):
    '''
    Creates the transpose of a matrix.
    param A: matrix in dimensions m by n
    return: transpose of a A (n by m)
    '''
    transpose = []
    for row in range(len(A[0])): # loop through matrix A rows
        transpose.append([])
        for col in range(len(A)): # loop through matrix A columns
            transpose[row].append(A[col][row]) # append new row with transpose elements

    return np.array(transpose) # return transpose matrix


def calcEigenvalues(A):
    '''
    Calculates all of the eigenvalues of a matrix.
    param A: matrix of dimension m by n
    return: list of eigenvalues in dessending order
    '''
    eigenvalues, Eigenvectors = np.linalg.eig(A)
    return np.sort(eigenvalues)[::-1]


def calcEigenvectors(A):
    '''
    Calculates all of the eigenvalues of a matrix
    param A: matrix of dimension m by n
    return: orthogonal matrix of eigenvalues
    '''
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    sorted_eigenvectors = []
    # Get index values for eigenvectors from unsorted eigenvalues vector
    for num in sorted_eigenvalues:
        index = np.argmax(eigenvalues == num)
        sorted_eigenvectors.append(transpose(eigenvectors)[index])
    return np.array(sorted_eigenvectors)


def calcSigma(A, eigenvalues):
    '''
    Creates Sigma matrix from the original matrix and eigenvalues.
    param A: matrix of dimension m by n
    param eigenvalues: list of eigenvalues in descending order
    return: Sigma matrix of dimension m by n
    '''
    Sigma = []
    for row in range(len(A)): # loop through matrix A rows
        Sigma.append([])
        for col in range(len(A[0])): # loop through matrix A columns
            Sigma[row].append(0) # append 0 to create Sigma matrix with each element set to 0

    for i in range(len(eigenvalues)): # loop through eigenvalues
        Sigma[i][i] = eigenvalues[i] ** 0.5 # set diagonals of Sigma to singular values

    return np.array(Sigma) # return Sigma matrix


def createSVD(A):
    '''
    Creates the SVD for A
    param A: matrix of dimension m by n
    return: matrix U (m by m)
    return: matrix Sigma (m by n)
    return: matrix V_transpose (n by n)
    '''
    # normalizing vectors for U
    norm_U = calcEigenvectors(multiply(A, transpose(A)))
    for row in range(len(norm_U)):
        accum = 0
        for col in range(len(norm_U[row])):
            accum += norm_U[row][col] ** 2
        norm = accum ** 0.5
        for col in range(len(norm_U[row])):
            norm_U[row][col] /= norm
    # normalizing vectors for V_T
    norm_V_T = calcEigenvectors(multiply(transpose(A), A))
    for row in range(len(norm_V_T)):
        accum = 0
        for col in range(len(norm_V_T[row])):
            accum += norm_V_T[row][col] ** 2
        norm = accum ** 0.5
        for col in range(len(norm_V_T[row])):
            norm_V_T[col] /= norm

    V_transpose = np.array(norm_V_T)
    Sigma = np.array(calcSigma(A, calcEigenvalues(multiply(transpose(A), A))))
    U = np.array(transpose(norm_U))

    return U, Sigma, V_transpose # return U matrix, Sigma matrix, V_T matrix


def createPCA(A, components):
    '''
    Create matrix of principal component vectors of matrix A.
    param A: matrix of dimension m by n
    param components: number of principal components desired
    return: matrix of principal components (components by n)
    '''

    # center the data
    center = []
    for row in range(len(A)):
        n_row = []
        mean = sum(A[row])/len(A[row])
        for col in range(len(A[0])):
            n_row.append(A[row][col] - mean)
        center.append(n_row)

    # calculate the covariance matrix
    covariance = []
    for row in range(len(center)):
        n_row = []
        for col in range(len(center)):
            accum = 0
            for k in range(len(center[0])):
                accum += center[row][k] * center[col][k]
            cov_num = accum / len(center[0])
            n_row.append(cov_num)
        covariance.append(n_row)

    # calculate SVD of covariance matrix
    U, Sigma, V_transpose = createSVD(covariance)

    return U[:components], Sigma # return U matrix


def trace(A):
    '''
    Find the trace of matrix A.
    '''
    accum = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == j:
                accum += A[i][j]
    return accum


def trace_from_singular_values(A, n):
    '''
        Find the trace of matrix A using the top n singular values.
    '''
    accum = 0
    for i in range(len(A)):
        for j in range(n):
            if i == j:
                accum += A[i][j]
    return accum


if __name__ == "__main__":
    main()
