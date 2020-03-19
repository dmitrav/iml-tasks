
import pandas, numpy
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

script_version = "v6"

if __name__ == "__main__":

    path = "/Users/andreidm/ETH/courses/iml-tasks/1/data/train_1b.csv"

    with open(path) as file:
        data = pandas.read_csv(file)

    y = data.iloc[:,1].values

    X_linear = data.iloc[:,2:].values
    X_quadratic = numpy.power(X_linear, 2)
    X_exponential = numpy.exp(X_linear)
    X_cosine = numpy.cos(X_linear)
    X_constant = numpy.full((700, 1), 1)

    X = numpy.hstack([X_linear, X_quadratic, X_exponential, X_cosine, X_constant])

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    results = []
    scores = []

    model = LinearRegression().fit(X, y)
    score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))

    scores.append(score)
    results.append({"model": "linear", "alpha": "-", "rmse": score, "coefs": model.coef_})
    print("linear score:", score)
    print("coefs:", model.coef_, "\n")

    for alpha in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]:

        model = Lasso(alpha=alpha).fit(X, y)
        score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))

        scores.append(score)
        results.append({"model": "lasso", "alpha": alpha, "rmse": score, "coefs": model.coef_})
        print("lasso alpha:", alpha, "score:", score)
        print("coefs:", model.coef_, "\n")

        model = Ridge(alpha=alpha).fit(X, y)
        score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))

        scores.append(score)
        results.append({"model": "ridge", "alpha": alpha, "rmse": score, "coefs": model.coef_})
        print("ridge alpha:", alpha, "score:", score)
        print("coefs:", model.coef_, "\n")

        model = ElasticNet(alpha=alpha).fit(X, y)
        score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))

        scores.append(score)
        results.append({"model": "elastic", "alpha": alpha, "rmse": score, "coefs": model.coef_})
        print("elastic alpha:", alpha, "score:", score)
        print("coefs:", model.coef_, "\n")

    print("min score:", min(scores))
    best_model_index = scores.index(min(scores))

    output = "\n".join([str(coef) for coef in results[best_model_index]["coefs"]])

    with open(path.replace("data/train_1b", "res/submission_1b_"+script_version), 'w') as file:
        file.write(output)

