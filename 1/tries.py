

import numpy
from sklearn.linear_model import LinearRegression

X = numpy.array([[1, 1], [1, 2], [2, 2], [2, 3]])

# y = 1 * x_0 + 2 * x_1 + 3
y = numpy.dot(X, numpy.array([1, 2])) + 3

reg = LinearRegression().fit(X, y)

print(reg.score(X, y))  # r2
print(reg.coef_)