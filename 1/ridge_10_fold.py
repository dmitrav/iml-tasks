
import pandas, numpy
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

script_version = "v.0.0.2"

if __name__ == "__main__":

    path = "/Users/dmitrav/ETH/courses/iml-tasks/1/data/train_1a.csv"

    with open(path) as file:
        data = pandas.read_csv(file)

    y = data.iloc[:,1]
    X = data.iloc[:,2:]

    kf = KFold(n_splits=10)

    rmse = []

    for alpha in [0.01, 0.1, 1, 10, 100]:

        model = Ridge(alpha=alpha)
        cv = KFold(n_splits=10, random_state=42)
        score = numpy.mean(cross_val_score(model, X, y, cv=10))

        print("alpha: ", alpha, ", score: ", score, sep="")

        rmse.append(score)

    output = "\n".join([str(score) for score in rmse])

    with open(path.replace("data/train_1a", "res/submission_1a"), 'w') as file:
        file.write(output)

