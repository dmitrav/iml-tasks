
import pandas, numpy, time, json, warnings
from project_2.src import preprocessing
from project_2.src.constants import train_path, train_labels_path
from project_2.src.constants import subtask_1_labels
from project_2.src.constants import version
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def create_svm_models(C_range, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    svm_models = []
    for C in C_range:
        model = ("svm_sigmoid_" + str(C), SVC(C=C, kernel="sigmoid", random_state=random_seed, probability=True))
        svm_models.append(model)

    return svm_models


def bruteforce():
    """ This method was meant to iterate over multiple fitting strategies.
        Turned out ot be infeasible timewise... """

    full_data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    data_with_nans = full_data.iloc[:, 3:]

    random_seeds = [42, 111, 666, 121, 321, 7, 26, 33, 222, 842]

    warnings.filterwarnings("ignore")

    start_time = time.time()

    features = preprocessing.get_engineered_features(numpy.array(data_with_nans))  # slow
    timepoint_1 = time.time()
    print(timepoint_1 - start_time, "s for feature engineering")

    imputed_features = preprocessing.impute_data_with_strategies(features)
    timepoint_2 = time.time()
    print(timepoint_2 - timepoint_1, "s for imputation\n")

    results = {"svm_models": []}

    for i in range(len(imputed_features)):

        timepoint_1 = time.time()
        print("working with:", imputed_features[i][0])

        # scaling takes < 5 seconds
        imputed_scaled_features = preprocessing.scale_data_with_methods(imputed_features[i][1])

        for j in range(len(imputed_scaled_features)):

            print("scaled by:", imputed_scaled_features[j][0], "\n")

            X = imputed_scaled_features[j][1]

            for n_folds in range(5, 21):
                for r in range(len(random_seeds)):

                    svm_models = create_svm_models([10 ** x for x in range(-6, 7)], random_seeds[r])
                    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seeds[r])

                    timepoint_1 = time.time()
                    print("models' evaluation started...")

                    # iterate over models
                    for model_name, model in svm_models:

                        result = {
                            "labels": [],
                            "scores": [],
                            "model": model_name,
                            "kfold": n_folds,
                            "random_seed": random_seeds[r],
                            "scaling": imputed_scaled_features[j][0],
                            "imputation": imputed_features[i][0],
                            "engineering": "median, min, max, var, unique",
                            "version": version
                        }

                        # iterate over labels to predict
                        for label in subtask_1_labels:
                            y = labels.loc[:, label]
                            mean_accuracy = cross_val_score(model, X, y, cv=cv)

                            result["labels"].append(label)
                            result["scores"].append(mean_accuracy)

                        results["svm_models"].append(result)

                    timepoint_2 = time.time()
                    print("results appended,", timepoint_2 - timepoint_1, "s elapsed\n")

                    iteration = i * len(imputed_scaled_features) + j * 15 + (n_folds - 5) * len(random_seeds) + r
                    total = len(imputed_features) * len(imputed_scaled_features) * 15 * len(random_seeds)
                    print(iteration / total, "% finished")

    # save main results
    with open("/Users/andreidm/ETH/courses/iml-tasks/project_2/res/svm_results_" + version + ".json", "w") as file:
        json.dump(results, file)


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    labels = pandas.read_csv(train_labels_path)

    # read precalculated featured
    features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/impute_simple_mean_v.0.0.8.csv"
    features = numpy.array(pandas.read_csv(features_path))

    # take a subset of features to train a model faster (10% of the entire dataset)
    indices = numpy.random.choice(features.shape[0], features.shape[0] // 10)
    features_subset = features[indices, :]
    labels = labels.iloc[indices, :]  # keep dataframe to be able to select label by name

    random_seeds = [42, 666, 321]

    results = {"svm_models": []}

    # scaling for full dataset takes < 5 seconds
    imputed_scaled_features = preprocessing.scale_data_with_methods(features_subset)

    for j in range(len(imputed_scaled_features)):

        print("scaled by:", imputed_scaled_features[j][0], "\n")

        X = imputed_scaled_features[j][1]

        for n_folds in range(5, 11):

            print("using", n_folds, "CV folds")

            for r in range(len(random_seeds)):

                print("with random seed:", r)

                svm_models = create_svm_models([10 ** x for x in range(-5, 6)], random_seeds[r])
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seeds[r])

                timepoint_1 = time.time()
                print("models' evaluation started...")

                # iterate over models
                for model_name, model in svm_models:

                    result = {
                        "labels": [],
                        "scores": [],
                        "model": model_name,
                        "kfold": n_folds,
                        "random_seed": random_seeds[r],
                        "scaling": imputed_scaled_features[j][0],
                        "imputation": "impute_simple_mean",
                        "engineering": "median, min, max, var, unique",
                        "version": version
                    }

                    # iterate over labels to predict
                    for label in subtask_1_labels:
                        y = labels.loc[:, label]
                        mean_accuracy = cross_val_score(model, X, y, cv=cv)

                        result["labels"].append(label)
                        result["scores"].append(mean_accuracy)

                    results["svm_models"].append(result)

                timepoint_2 = time.time()
                print("results appended,", timepoint_2 - timepoint_1, "s elapsed")

                iteration = j * 5 + (n_folds - 5) * len(random_seeds) + r
                total = len(imputed_scaled_features) * 5 * len(random_seeds)
                print(iteration / total * 100, "% finished\n")

    # save main results
    with open("/Users/andreidm/ETH/courses/iml-tasks/project_2/res/results_svm_impute_simple_mean_" + version + ".json", "w") as file:
        json.dump(results, file)
