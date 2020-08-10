"""
 Dimensionality_Reduction - Example
 (Using PCA)
"""
import numpy as np

class Dimensionality_reduction_ex():
    def __init__(self):
        # data: an 4x2 matrix
        self.arr = np.array(
                [ [1,2],
                  [-2,-3.5],
                  [3,5],
                  [-4,-7]
                ]
            )
        
    # 1. feature noralization
    def standardize(self):
        mean_colVector = np.mean(self.arr, axis=0)
        self.standard_arr = self.arr - mean_colVector
    
    # 2. calculate covariance matrix
    def cal_Cov_Matrix(self):
        self.Cov_Matrix = np.cov(self.standard_arr, rowvar=0)
    
    # 3. calculate eigenvalues and eigenvectors for covariance matrix 
    def cal_eigVals_and_eigVecs(self):
        self.eigVals, self.eigVecs = np.linalg.eig(self.Cov_Matrix)
    
    # 4. select members of eigenvectors according to
    #    top k largest eigenvalues, and then
    #    obtain the eigenvector-matrix
    def get_top_k_matrix_of_eigVecs(self, k):
        sorted_eigVals = np.argsort(self.eigVals)
        k_top_eigVals = sorted_eigVals[:-(k+1):-1]
        self.eigenvector_matrix = self.eigVecs[:, k_top_eigVals]
    
    # 5. get the final dimensionality-reduced result 
    #    by dot product operation between 
    #    original matrix and eigenvector-matrix
    def get_result(self):
        self.result = np.dot(self.standard_arr, self.eigenvector_matrix)
        
if __name__ == "__main__":
    ex = Dimensionality_reduction_ex()
    # an 4x2 matrix (we want to reduce the dimension)
    print("original array:\n", ex.arr, "\n")
    
    ex.standardize()
    print("standardized array:\n", ex.standard_arr, "\n")

    ex.cal_Cov_Matrix()
    print("covariance matrix:\n", ex.Cov_Matrix, "\n")
    
    ex.cal_eigVals_and_eigVecs()
    print("eiaenvalues:\n", ex.eigVals, "\n")
    print("eigenvectors:\n", ex.eigVecs, "\n")
    
    # obtain an new 2x1 matrix
    ex.get_top_k_matrix_of_eigVecs(k=1)
    print("eigenvector matrix:\n", ex.eigenvector_matrix, "\n")
    
    # the result turns to be an 4x1 matrix (cause (4x2) x (2x1) => (4x1))
    ex.get_result()
    print("dimensionality-reduced result:\n", ex.result)
    
    