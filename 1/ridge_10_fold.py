
import pandas, numpy
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.linear_model import Ridge

script_version = "v.0.0.3"

if __name__ == "__main__":

    path = "/Users/dmitrav/ETH/courses/iml-tasks/1/data/train_1a.csv"

    with open(path) as file:
        data = pandas.read_csv(file)

    y = data.iloc[:,1]
    X = data.iloc[:,2:]

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    rmse = []

    for alpha in [0.01, 0.1, 1, 10, 100]:

        model = Ridge(alpha=alpha)

        score = numpy.sqrt(-numpy.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)))

        print("alpha: ", alpha, ", score: ", score, sep="")

        rmse.append(score)

    output = "\n".join([str(score) for score in rmse])

    with open(path.replace("data/train_1a", "res/submission_1a"), 'w') as file:
        file.write(output)

