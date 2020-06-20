from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# dataset
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

# fit a k-nearest neighbor model to the data
K = 3
model = KNeighborsClassifier(n_neighbors = K)
model.fit(X, y)
print(model)

# make predictions
print( '(-2, -2, 2, 2) is class')
print( model.predict([[-2,-2, 2, 2]]) ) # class 0

print( '(1, 5, 5, 1) is class'),
print( model.predict([[1, 5, 5, 1]]) ) # class 1

print( '(10, 10, 10, 10) is class')
print( model.predict([[10, 10, 10, 10]]) ) # class 2
