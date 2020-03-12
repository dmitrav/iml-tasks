
import pandas, numpy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

script_version = "v.0.0.1"

if __name__ == "__main__":

    path = "/Users/dmitrav/ETH/courses/iml-tasks/1/data/train_1b.csv"

    with open(path) as file:
        data = pandas.read_csv(file)

    y = data.iloc[:,1].values
    X_linear = data.iloc[:,2:].values
    X_quadratic = numpy.power(X_linear, 2)
    X_exponential = numpy.exp(X_linear)
    X_cosine = numpy.cos(X_linear)
    X_constant = numpy.full(X_linear.shape, 1)

    model = LinearRegression()

    coefs = []

    model.fit(X_linear, y)
    coefs.extend(model.coef_)
    print("linear r2:", model.score(X_linear, y))
    print("coefs:", model.coef_, "\n")

    model.fit(X_quadratic, y)
    coefs.extend(model.coef_)
    print("quadratic r2:", model.score(X_quadratic, y))
    print("coefs:", model.coef_, "\n")

    model.fit(X_exponential, y)
    coefs.extend(model.coef_)
    print("exponential r2:", model.score(X_exponential, y))
    print("coefs:", model.coef_, "\n")

    model.fit(X_cosine, y)
    coefs.extend(model.coef_)
    print("cosine r2:", model.score(X_cosine, y))
    print("coefs:", model.coef_, "\n")

    model.fit(X_constant, y)
    coefs.append(0.)
    print("constant r2:", model.score(X_constant, y))
    print("coefs:", model.coef_)

    output = "\n".join([str(coef) for coef in coefs])

    with open(path.replace("data/train_1b", "res/submission_1b"), 'w') as file:
        file.write(output)

