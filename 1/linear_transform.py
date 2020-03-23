
import pandas, numpy
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import mutual_info_regression, f_regression
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import pearsonr, spearmanr

script_version = "v14"

if __name__ == "__main__":

    path = "/Users/andreidm/ETH/courses/iml-tasks/1/data/train_1b.csv"

    with open(path) as file:
        data = pandas.read_csv(file)

    y = data.iloc[:,1].values
    X_linear = data.iloc[:,2:].values

    # # filtering the data
    # for i in range(X_linear.shape[1]):
    #
    #     q1 = numpy.quantile(X_linear[:,i], 0.25)
    #     q3 = numpy.quantile(X_linear[:,i], 0.75)
    #     iqr = q3 - q1
    #
    #     filter = (X_linear[:,i] >= q1 - 1.5 * iqr) * (X_linear[:,i] <= q3 + 1.5 * iqr)
    #
    #     X_linear = X_linear[filter, :]
    #     y = y[filter]

    # # visualising the data
    # df = pandas.DataFrame(X_linear)
    # df.columns = ["x1", "x2", "x3", "x4", "x5"]
    #
    # df = df.melt(var_name='vars', value_name='vals')
    #
    # ax = seaborn.boxplot(x="vars", y="vals", data=df)
    # plt.show()

    X_quadratic = numpy.power(X_linear, 2)
    X_exponential = numpy.exp(X_linear)
    X_cosine = numpy.cos(X_linear)
    X_constant = numpy.full((X_linear.shape[0], 1), 1)

    X = numpy.hstack([X_linear, X_quadratic, X_exponential, X_cosine, X_constant])

    # corrs = [spearmanr(X[:,i], y)[0] for i in range(X.shape[1])]  # none
    # mics = mutual_info_regression(X, y, discrete_features=False, n_neighbors=10, random_state=42)  # none
    # f_scores = f_regression(X, y, center=False)  # questionable

    folds = [3, 4, 5]

    alphas = numpy.power(10, numpy.linspace(-6, 6, 1000))  # v. 13
    # alphas = [1e-6, 1e-5, 1e-4, 1e-4, 1e-3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]  # v.9-12
    # alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]  # v.8

    results = []
    scores = []

    for fold in folds:

        cv = KFold(n_splits=fold, shuffle=True, random_state=42)

        model = LinearRegression().fit(X, y)
        # score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))  # v.1-12
        score = numpy.sum(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))  # v. 14

        scores.append(score)
        results.append({"model": "linear", "fold": fold, "alpha": "-", "rmse": score, "coefs": model.coef_})
        # print("linear score:", score)
        # print("coefs:", model.coef_, "\n")

        for alpha in alphas:

            model = Lasso(alpha=alpha).fit(X, y)
            # score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
            score = numpy.sum(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))  # v. 14

            scores.append(score)
            results.append({"model": "lasso", "fold": fold, "alpha": alpha, "rmse": score, "coefs": model.coef_})
            # print("lasso alpha:", alpha, "fold:", fold, "score:", score)
            # print("coefs:", model.coef_, "\n")

            model = Ridge(alpha=alpha).fit(X, y)
            # score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
            score = numpy.sum(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))  # v. 14

            scores.append(score)
            results.append({"model": "ridge", "fold": fold, "alpha": alpha, "rmse": score, "coefs": model.coef_})
            # print("ridge alpha:", alpha, "fold:", fold, "score:", score)
            # print("coefs:", model.coef_, "\n")

            model = ElasticNet(alpha=alpha).fit(X, y)
            # score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))
            score = numpy.sum(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))  # v. 14

            scores.append(score)
            results.append({"model": "elastic", "fold": fold, "alpha": alpha, "rmse": score, "coefs": model.coef_})
            # print("elastic alpha:", alpha, "fold:", fold, "score:", score)
            # print("coefs:", model.coef_, "\n")

    print("min score:", min(scores))
    best_model_index = scores.index(min(scores))

    print(results[best_model_index])

    output = "\n".join([str(coef) for coef in results[best_model_index]["coefs"]])

    with open(path.replace("data/train_1b", "res/submission_1b_"+script_version), 'w') as file:
        file.write(output)

