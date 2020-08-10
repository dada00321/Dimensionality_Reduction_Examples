import numpy as np
from sklearn import datasets

class PCA_test():
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        self.data = datasets.load_iris()["data"]
        
    # 1. feature noralization
    def standardize(self):
        self.mean_colVector = np.mean(self.data, axis=0)
        self.standard_arr = self.data - self.mean_colVector
    
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
        sorted_eigVals = np.argsort(self.eigVals)[::-1] # descending order
        k_top_eigVals = sorted_eigVals[:k] # top k largest eigenvalues
        self.eigenvector_matrix = self.eigVecs[:, k_top_eigVals]
    
    # 5. get the final dimensionality-reduced result 
    #    by dot product operation between 
    #    original matrix and eigenvector-matrix
    def get_result(self):
        self.result = np.dot(self.standard_arr, self.eigenvector_matrix)

if __name__ == "__main__":
    ex = PCA_test()
    # an 4x2 matrix (we want to reduce the dimension)
    print("original array:\n", ex.data, "\n")
    
    ex.standardize()
    print("standardized array:\n", ex.standard_arr, "\n")

    ex.cal_Cov_Matrix()
    print("covariance matrix:\n", ex.Cov_Matrix, "\n")
    
    ex.cal_eigVals_and_eigVecs()
    print("eiaenvalues:\n", ex.eigVals, "\n")
    print("eigenvectors:\n", ex.eigVecs, "\n")
    
    # obtain an new 2x1 matrix
    ex.get_top_k_matrix_of_eigVecs(k=2)
    print("eigenvector matrix:\n", ex.eigenvector_matrix, "\n")
    
    ex.get_result()
    print("dimensionality-reduced result:\n", ex.result)
    
    print("="*35)
    print("dim. of original data:", ex.data.shape)
    print("\nAfter dimensionality reduction...\n")
    print("dim. of result:", ex.result.shape)
    
    