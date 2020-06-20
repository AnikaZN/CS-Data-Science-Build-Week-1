import numpy as np
from scipy import stats
from sklearn import datasets


class K_Nearest_Neighbors():
    """
    Determine the classifcation of a given point based on existing data
    points
    Takes in number of neighbors to be calculated with
    """
    def __init__(self, k):
        self.k = k

    def euclidean_distance(self, vectorA, vectorB):
        """
        Helper function to calculate Euclidean distance
        """
        a = vectorA[0] - vectorB[0]
        b = vectorA[1] - vectorB[1]
        a2 = a ** 2
        b2 = b ** 2
        distance = np.sqrt(a2 + b2)
        return distance

    def fit(self, X, y):
        """
        X is a list of lists containing x and y values
        y is a list of classifications
        """
        for i in range(len(X)):
            X[i].append(y[i])
        return X

    def plot(self, X, target):
        """
        Plotting
        Two-dimensional plots ONLY
        """
        # TODO
        pass

    def predict(self, X, target):
        """
        Target is a tuple
        """
        for point in X:
            distance = self.euclidean_distance(target, point)
            point.append(distance)

        sorted_data = sorted(X, key=lambda data: data[-1])
        top_x = sorted_data[0:self.k]

        array = []
        for row in top_x:
            classification = row[-2]
            array.append(classification)

        object = stats.mode(array)
        mode = object.mode[0]

        for point in X:
            del point[-1]

        return mode


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data.tolist()
    y = iris.target.tolist()

    model = K_Nearest_Neighbors(3)
    model.fit(X, y)

    # Make predictions
    print('(-2, -2, 2, 2) is class:')
    print(model.predict(X, [-2, -2, 2, 2]))  # Class 0

    print('(1, 5, 5, 1) is class:')
    print(model.predict(X, [1, 5, 5, 1]))  # Class 0

    print('(10, 10, 10, 10) is class:')
    print(model.predict(X, [10, 10, 10, 10]))  # Class 2
