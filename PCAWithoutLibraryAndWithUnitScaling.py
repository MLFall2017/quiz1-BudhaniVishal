import pandas as pd
import numpy as np

# Mean Calculation Function
def calculateMean(list):
    average = 0
    sum = 0
    for n in list:
        sum = sum + n
    average = sum / len(list)
    return average
# Mean function closed

#Calculate Variance
def calculateVariance(list, mean):
    sum = 0
    variance = 0
    for n in list:
        sum = sum + ((n-mean)**2)
    variance = sum/(len(list)-1)
    return variance
# CalculateVariance




dataFrame = pd.read_csv('C:\MS\Fall2017\MachineLearning\Quiz\dataset_1.csv')
#print the column names
print("Printing Columns")
print (dataFrame.columns)
#print the column names

dataFrame.columns=['x', 'y', 'z']
dataFrame.dropna(how="all", inplace=True) # drops the empty line at file-end
print(dataFrame.tail())

x_value = dataFrame.ix[:,0]
print(x_value.tail())
print("Variance of x is")
print(calculateVariance(x_value, calculateMean(x_value)))
print(np.var(x_value))

# Taken all the values in excel
X = dataFrame.ix[:,0:4].values
print("Printing x")
print(X)
# Taken all the values in Excel

# let us continue with the transformation of the data onto unit scale (mean=0 and variance=1),
# which is a requirement for the optimal performance of many machine learning algorithms.
from sklearn.preprocessing import StandardScaler
standardValue = StandardScaler().fit_transform(X)
print("Printing standard_Val")
print(standardValue)
# Standardizing the data

# Calculating Covariance Matrix
mean_vec = np.mean(standardValue, axis=0)
print("Standard Mean_Vector")
print(mean_vec)

covarianceMatrix = (standardValue - mean_vec).T.dot((standardValue - mean_vec)) / (standardValue.shape[0]-1)
print('Covariance matrix \n%s' %covarianceMatrix)
# Calculating Covariance Matrix

# Performing eigen decomposition on covariance matrix
eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)

print('Eigen Vectors are as \n%s' % eigenVectors)
print('\nEigen Values are as \n%s' % eigenValues)

# Perform an eigendecomposition on the covariance matrix

# Asserting
for ev in eigenVectors:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok! \n')
#Asserting

# Make a list of (eigenvalue, eigenvector) tuples
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:, i]) for i in range(len(eigenValues))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigenPairs.sort()
eigenPairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for item in eigenPairs:
    print(item[0])

# Reducing the dimensionality
dimensionalReducedMatrix = np.hstack((eigenPairs[0][1].reshape(3, 1),
                      eigenPairs[1][1].reshape(3, 1)))

print('Matrix W:\n', dimensionalReducedMatrix)
# Reducing the dimensionality

# In this last step we will use the 4*24*2-dimensional projection matrix
# WW to transform our samples onto the new subspace via the equation
# Y=X*W where Y is a 1000*2 matrix of our transformed samples.

finalReduced = standardValue.dot(dimensionalReducedMatrix)
print('Printing the final value \n', finalReduced)


final1 = finalReduced[:,0]
final2 = finalReduced[:,1]

#print('\n Final1:\n', final1)
#print('\n Final2:\n', final2)

import matplotlib.pyplot as plt
plt.interactive(False)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(final1, final2)
plt.show()


#Added to validate the answer
#Library Implementation
#from sklearn.decomposition import PCA as sklearnPCA
#sklearn_pca = sklearnPCA(n_components=2)
#Y_sklearn = sklearn_pca.fit_transform(standardValue)
#print('Library Value')
#print(Y_sklearn)
#Library Implementation
