import numpy as np
from scipy import stats

class K_Nearest_Neighbors():
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

    def predict(self, X, target):
        """
        Target is a tuple
        """
        for point in X:
            distance = self.euclidean_distance(target, point)
            point.append(distance)

        sorted_data = sorted(X, key=lambda data: data[3])
        top_x = sorted_data[0:self.k]

        array = []
        for row in top_x:
            classification = row[2]
            array.append(classification)
            object = stats.mode(array)
            mode = object.mode[0]
        return mode


# X = [
#     [1.465489372,2.362125076],
#     [3.396561688,4.400293529],
#     [1.38807019,1.850220317],
#     [3.06407232,3.005305973],
#     [7.627531214,2.759262235],
#     [5.332441248,2.088626775],
#     [6.922596716,1.77106367],
#     [8.675418651,-0.242068655],
#     [7.673756466,3.508563011]
#     ]
# y = [0, 0, 0, 0, 1, 1, 1, 1, 1]
# point = [2.7810836,2.550537003]


X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
y =  [0, 0, 0, 1, 1, 1]

point1 = [-2, -2]
point2 = [3, 2]


if __name__ == '__main__':
    model = K_Nearest_Neighbors(3)
    model.fit(X, y)
    print(model.predict(X, point1))
    print(model.predict(X, point2))
