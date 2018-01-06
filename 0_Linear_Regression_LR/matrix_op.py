% For linear regression models that use least squares method, 
% you can generate as many data columns as you like. However,
% you still need to eliminate any dependent columns to ensure
% that your input matrix is full column rank. The matrix 
% operations found in this scripts focus on elimating dependent
% columns, reshaping matrices, and adding new columns dependent 
% on the input matrix.

import numpy.matlib
import numpy as np

def column(matrix, i): %extracts column from a matrix
    return [row[i] for row in matrix]

def full_col_rank(matrix): %checks and eliminates any dependant columns.
    index_dep = [] %List to store dependent column indices
    for i in range(len(matrix[0,:])):
        col_i = column(matrix, i) 
        mat_i = np.matmul(matrix.transpose(),col_i)
        for j in range(len(mat_i[:,0])):
            ref_j = mat_i[j,0]
            sum_j = 0
            for element in mat_i[j,:]:
                if ref_j - element <= 0.01:
                    sum_j = sum_j +1
            if sum_j == matrix.shape[1]:
                index_dep.append(i)
    if index_dep:
        mod_mat_index = 0
        mod_mat = np.zeros((len(matrix[:,0]),len(matrix[0,:])-len(index_dep)[0]))
        for i in range(len(matrix[0,:])):
            if i not in index_dep:
                for j in range(len(matrix[:,0])):
                    mod_mat[j,mod_mat_index] = matrix[j,i]
                mod_mat_index = mod_mat_index + 1 
    else:
        mod_mat = matrix
    return mod_mat
