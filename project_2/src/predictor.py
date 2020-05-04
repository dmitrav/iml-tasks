
import pandas, numpy, time, json, warnings, multiprocessing
from project_2.src import preprocessing
from project_2.src.constants import train_path, train_labels_path
from project_2.src.constants import subtask_1_labels
from project_2.src.constants import version
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek


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


def downscaled_bruteforce():
    """ Downscaled method used in v.0.0.8.
        Kept unchanged since then. """

    warnings.filterwarnings("ignore")

    labels = pandas.read_csv(train_labels_path)

    # read precalculated featured
    features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/impute_simple_mean_v.0.0.8.csv"
    features = numpy.array(pandas.read_csv(features_path))

    # take a subset of features to train a model faster (30% of the entire dataset)
    indices = numpy.random.choice(features.shape[0], features.shape[0] // 3)
    features_subset = features[indices, :]
    labels = labels.iloc[indices, :]  # keep dataframe to be able to select label by name

    random_seeds = [42]
    n_folds = [5]

    results = {"svm_models": []}

    # scaling for full dataset takes < 5 seconds
    imputed_scaled_features = preprocessing.scale_data_with_methods(features_subset)

    for j in range(len(imputed_scaled_features)):

        print("scaled by:", imputed_scaled_features[j][0], "\n")

        X = imputed_scaled_features[j][1]
        timepoint_1 = time.time()

        for k in range(len(n_folds)):

            print("using", n_folds[k], "CV folds")

            for r in range(len(random_seeds)):

                print("with random seed:", random_seeds[r])

                svm_models = create_svm_models([10 ** x for x in range(-5, 6)], random_seeds[r])
                cv = KFold(n_splits=n_folds[k], shuffle=True, random_state=random_seeds[r])

                print("models' evaluation started...")

                # iterate over models
                for m in range(len(svm_models)):

                    # print("\n", model_name, ":\n", sep="")

                    result = {
                        "labels": [],
                        "scores": [],
                        "model": svm_models[m][0],  # model name
                        "kfold": n_folds[k],
                        "random_seed": random_seeds[r],
                        "scaling": imputed_scaled_features[j][0],
                        "imputation": "impute_simple_mean",
                        "engineering": "median, min, max, var, unique",
                        "version": version
                    }

                    # iterate over labels to predict
                    for label in subtask_1_labels:
                        y = labels.loc[:, label]
                        f1_score = cross_val_score(svm_models[m][1], X, y, cv=cv, scoring='f1_weighted')

                        print(label, "scored with median f1 =", numpy.median(f1_score))

                        result["labels"].append(label)
                        result["scores"].append(f1_score.tolist())

                    print(round((m + 1) / len(svm_models) * 100, 2), "% of models scored\n")
                    results["svm_models"].append(result)

        timepoint_2 = time.time()
        print("results appended,", (timepoint_2 - timepoint_1) // 60 + 1, "minutes elapsed")
        print(round((j + 1) / len(imputed_scaled_features) * 100, 2), "% of the total run finished")

        # iteration = j * len(nfolds_range) + k * len(random_seeds) + r+1
        # total = len(imputed_scaled_features) * 5 * len(random_seeds)
        # print(round(iteration / total * 100, 2), "% finished\n")

    # save main results
    with open("/Users/andreidm/ETH/courses/iml-tasks/project_2/res/results_svm_impute_simple_mean_" + version + ".json",
              "w") as file:
        json.dump(results, file)


def run_label_specific_svm(label_name, imputation_name):

    seed = 42
    kfold = 10
    cv = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    svm_models = create_svm_models([10 ** x for x in range(-3, 1)], seed)
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'f1': 'f1'}

    folder = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/label_specific/flattened/"
    ending = "_v.0.0.27.csv"

    all_results = {"svm": []}

    labels = pandas.read_csv(train_labels_path)
    features = pandas.read_csv(folder + label_name + "_" + imputation_name + ending)
    labels = labels.loc[labels.loc[:, "pid"].isin(features["pid"]), label_name]

    # take a subset of features to train a model faster (of size shape[0] / factor)
    if features.shape[0] > 15000:
        factor = 3
    elif 10000 < features.shape[0] <= 15000:
        factor = 2
    else:
        factor = 1

    indices = numpy.random.choice(features.shape[0], features.shape[0] // factor)
    features_subset = features.iloc[indices, :]
    labels_subset = labels.iloc[indices]  # keep dataframe to be able to select label by name

    scaled_features = preprocessing.scale_data_with_methods(features_subset.iloc[:, 2:])

    for j in range(len(scaled_features)):

        # print("with scaling: ", scaled_features[j][0])

        # print("balancing data... ", end="")
        resampler = SMOTETomek(random_state=seed)
        X_resampled, y_resampled = resampler.fit_resample(scaled_features[j][1], labels_subset)
        # print("done!")

        for i in range(len(svm_models)):

            # print("evaluation for", svm_models[i][0], "started...")

            timepoint_1 = time.time()

            scores = cross_validate(svm_models[i][1], X_resampled, y_resampled, cv=cv, scoring=scoring)

            print(svm_models[i][0], "for", label_name, "with imputation", imputation_name, "and scaling", scaled_features[j][0], "scored with:")
            for key in scoring.keys():
                print(key, "=", scores["test_" + key])
            print()

            timepoint_2 = time.time()
            print(int((timepoint_2 - timepoint_1) // 60 + 1), "minutes elapsed\n")

            all_results["svm"].append({
                "labels": label_name,
                "scores": {
                    'accuracy': scores["test_accuracy"].tolist(),
                    'precision': scores["test_precision"].tolist(),
                    'recall': scores["test_recall"].tolist(),
                    'roc_auc': scores["test_roc_auc"].tolist(),
                    'f1': scores["test_f1"].tolist()
                },
                "model": svm_models[i][0],  # model name
                "kfold": kfold,
                "random_seed": seed,
                "scaling": scaled_features[j][0],
                "imputation": imputation_name,
                "engineering": "median, min, max, var, finites, sum, slope",
                "version": version
            })

    # save results
    outfile = "/Users/andreidm/ETH/courses/iml-tasks/project_2/res/results_flattened_" + label_name + "_" + imputation_name + "_" + version + ".json"
    with open(outfile, "w") as file:
        json.dump(all_results, file)


if __name__ == "__main__":

    processes = []
    start_time = time.time()

    for imputation in ["impute_iter_mean"]:
        for label in ['LABEL_Lactate',  'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_SaO2']:

            p = multiprocessing.Process(target=run_label_specific_svm, args=(label,imputation))
            processes.append(p)
            p.start()

    for process in processes:
        process.join()

    print('All done within', int((time.time() - start_time) // 3600 + 1), "hours")