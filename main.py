# Gabe Galindo
# CYBR 304 - Foundation of Computational Mathematics
# Final Project - K-Means Clustering
# 05/15/2025

import helper as mat
import pandas as pd
import matplotlib.pylab as plt

def main():
    # load csv and convert to numpy array
    data_matrix =  mat.np.array(pd.read_csv("fwpHarvestEstimatesReport.csv"))
    convert_N_R(data_matrix) # convert residency values to integers

    # call createPCA method from helper
    U_t, Sigma = mat.createPCA(mat.transpose(data_matrix), 2)
    U = mat.transpose(U_t)
    print(f"First two Principal Components of U:\n{U}\n")

    # find significance of principal components
    trace_Sigma = mat.trace(Sigma)
    components_trace = mat.trace_from_singular_values(Sigma, 2)
    print(f"Significance of the PCA: {components_trace / trace_Sigma:.2%}\n")

    # project data onto principal components
    project = mat.multiply(data_matrix, U)

    # plot on scatter plot
    x = mat.transpose(project)[0]
    y = mat.transpose(project)[1]
    plt.scatter(x, y)

    # sort sample row indexes into clusters based on x values
    c1, c2, c3, c4 = [], [], [], []
    count = 2
    for i in range(len(x)):
        if x[i] < -112:
            c1.append(count)
        elif x[i] < -68:
            c2.append(count)
        elif x[i] < -36:
            c3.append(count)
        else:
            c4.append(count)
        count += 1
    print("Clustered row indexes:")
    print("Cluster 1:", c1)
    print("Cluster 2:", c2)
    print("Cluster 3:", c3)
    print("Cluster 4:", c4)

    plt.show() # show the scatter plot


def convert_N_R(data_matrix):
    '''
    Convert Residency values 'N' and 'R' to integers and return data matrix
    '''
    N_R_index = 1
    for i in range(len(data_matrix)):
        if data_matrix[i][N_R_index] == 'N':
            data_matrix[i][N_R_index] = 0
        elif data_matrix[i][N_R_index] == 'R':
            data_matrix[i][N_R_index] = 1


if __name__ == '__main__':
    main()