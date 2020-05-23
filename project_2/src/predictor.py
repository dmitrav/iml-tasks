
import pandas, numpy, time, json, warnings, multiprocessing
from project_2.src import preprocessing
from project_2.src.constants import train_path, train_labels_path, test_path
from project_2.src.constants import subtask_1_labels, subtask_2_labels, subtask_3_labels, version
from project_2.src.constants import version
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC, LinearSVR
from imblearn.combine import SMOTETomek
from sklearn import linear_model
from tqdm.auto import trange
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer


def create_svm_models(C_range, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    models = []
    for C in C_range:
        model = ("svm_sigmoid_" + str(C), SVC(C=C, kernel="sigmoid", random_state=random_seed, probability=True))
        models.append(model)

    return models


def create_svr_models(C_range, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    models = []
    for C in C_range:
        model = ("svr_" + str(C), LinearSVR(C=C, random_state=random_seed))
        models.append(model)

    return models


def create_sgd_models(alpha, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    models = []
    for loss in ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']:
        for a in alpha:
            model = ("sgd_" + loss + str(a), linear_model.SGDRegressor(loss=loss, alpha=a, random_state=random_seed))
            models.append(model)

    return models


def create_lasso_models(alpha, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    models = []
    for a in alpha:
        model = ("lasso_" + str(a), linear_model.Lasso(alpha=a, random_state=random_seed))
        models.append(model)

    return models


def create_ridge_models(alpha, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    models = []
    for a in alpha:
        model = ("ridge_" + str(a), linear_model.Ridge(alpha=a, random_state=random_seed))
        models.append(model)

    return models


def create_elastic_net_models(alpha, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    models = []
    for a in alpha:
        model = ("elastic_net_" + str(a), linear_model.ElasticNet(alpha=a, random_state=random_seed))
        models.append(model)

    return models


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
    svm_models = create_svm_models([10 ** x for x in range(-2, 1)], seed)
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'f1': 'f1'}

    folder = "/Users/dmitrav/ETH/courses/iml-tasks/project_2/data/label_specific/flattened/"
    ending = "_v.0.0.31.csv"

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

    indices = numpy.random.choice(features.shape[0], features.shape[0] // factor, replace=False)
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
    outfile = "/Users/dmitrav/ETH/courses/iml-tasks/project_2/res/subtask_2/results_" + label_name + "_" + imputation_name + "_" + version + ".json"
    with open(outfile, "w") as file:
        json.dump(all_results, file)


def run_label_specific_regression(label_name, imputation_name):

    seed = 415
    kfold = 10

    cv = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    alphas = [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5.0, 10, 50.0, 100, 500.0, 100]

    models = [# *create_sgd_models([10 ** x for x in range(-3, 4)], seed),
              # *create_svr_models([10 ** x for x in range(-3, 4)], seed),
              *create_lasso_models(alphas, seed),
              *create_ridge_models(alphas, seed),
              *create_elastic_net_models(alphas, seed)
              ]

    scoring = {'max_error': 'max_error',
               'neg_mean_squared': 'neg_mean_squared_error',
               'neg_median_absolute': 'neg_median_absolute_error',
               'r2': 'r2'}

    folder = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/label_specific/flattened/"
    ending = "_v.0.0.37.csv"

    all_results = {"results": []}

    labels = pandas.read_csv(train_labels_path)
    features = pandas.read_csv(folder + label_name + "_" + imputation_name + ending)
    labels = labels.loc[labels.loc[:, "pid"].isin(features["pid"]), label_name]

    # # take a subset of features to train a model faster (of size shape[0] / factor)
    # if features.shape[0] > 15000:
    #     factor = 2
    # elif 10000 < features.shape[0] <= 15000:
    #     factor = 1
    # else:
    #     factor = 1

    # regression take much faster, no need to subset
    factor = 1

    indices = numpy.random.choice(features.shape[0], features.shape[0] // factor, replace=False)
    features_subset = features.iloc[indices, :]
    labels_subset = labels.iloc[indices]  # keep dataframe to be able to select label by name

    scaled_features = preprocessing.scale_data_with_methods(features_subset.iloc[:, 2:])

    for j in range(len(scaled_features)):

        # print("with scaling: ", scaled_features[j][0])

        for i in range(len(models)):

            # print("evaluation for", svm_models[i][0], "started...")

            timepoint_1 = time.time()

            scores = cross_validate(models[i][1], scaled_features[j][1], labels_subset, cv=cv, scoring=scoring)

            print(models[i][0], "for", label_name, "with imputation", imputation_name, "and scaling", scaled_features[j][0], "scored with:")
            for key in scoring.keys():
                print(key, "=", scores["test_" + key])
            print()

            timepoint_2 = time.time()
            print(int((timepoint_2 - timepoint_1) // 60 + 1), "minutes elapsed\n")

            all_results["results"].append({
                "labels": label_name,
                "scores": {
                    'max_error': scores['test_max_error'].tolist(),
                    'neg_mean_squared': scores["test_neg_mean_squared"].tolist(),
                    'neg_median_absolute': scores["test_neg_median_absolute"].tolist(),
                    'r2': scores["test_r2"].tolist()
                },
                "model": models[i][0],  # model name
                "kfold": kfold,
                "random_seed": seed,
                "scaling": scaled_features[j][0],
                "imputation": imputation_name,
                "engineering": "median, min, max, var, finites, sum, slope",
                "version": version
            })

    # save results
    outfile = "/Users/andreidm/ETH/courses/iml-tasks/project_2/res/subtask_3/results_flattened_" + label_name + "_" + imputation_name + "_" + version + ".json"
    with open(outfile, "w") as file:
        json.dump(all_results, file)


def train_model_and_predict_on_test(label):

    labels = pandas.read_csv(train_labels_path)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    classification_scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'f1': 'f1'}
    regression_scoring = {'max_error': 'max_error', 'neg_mean_squared': 'neg_mean_squared_error', 'neg_median_absolute': 'neg_median_absolute_error', 'r2': 'r2'}

    if label == 'LABEL_BaseExcess':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_BaseExcess_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_BaseExcess_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_Fibrinogen':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Fibrinogen_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.01, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_Fibrinogen_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_AST':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_AST_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_AST_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_Alkalinephos':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Alkalinephos_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_Alkalinephos_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_Bilirubin_total':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Bilirubin_total_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_Bilirubin_total_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_Lactate':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Lactate_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_Lactate_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_TroponinI':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_TroponinI_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_TroponinI_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_SaO2':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_SaO2_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.01, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_SaO2_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_Bilirubin_direct':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Bilirubin_direct_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_Bilirubin_direct_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_EtCO2':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_EtCO2_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_EtCO2_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    # subtask 2
    elif label == 'LABEL_Sepsis':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Sepsis_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        resampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(train_data.iloc[:, 2:], labels[label])

        model = SVC(C=0.1, kernel="sigmoid", random_state=42, probability=True)
        model.fit(X_resampled, y_resampled)

        scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=classification_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in classification_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_Sepsis_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict_proba(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    # subtask 3
    elif label == 'LABEL_RRate':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_RRate_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        model = linear_model.Ridge(alpha=10, random_state=415)
        model.fit(train_data.iloc[:, 2:], labels[label])

        scores = cross_validate(model, train_data.iloc[:, 2:], labels[label], cv=cv, scoring=regression_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in regression_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_RRate_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_ABPm':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_ABPm_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        model = linear_model.Lasso(alpha=0.0005, random_state=415)
        model.fit(train_data.iloc[:, 2:], labels[label])

        scores = cross_validate(model, train_data.iloc[:, 2:], labels[label], cv=cv, scoring=regression_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in regression_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_ABPm_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_SpO2':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_SpO2_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        model = linear_model.Lasso(alpha=0.01, random_state=415)
        model.fit(train_data.iloc[:, 2:], labels[label])

        scores = cross_validate(model, train_data.iloc[:, 2:], labels[label], cv=cv, scoring=regression_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in regression_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_SpO2_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    elif label == 'LABEL_Heartrate':

        path_to_train = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Heartrate_train_imputed_v.0.0.47.csv'
        train_data = pandas.read_csv(path_to_train)

        labels = labels.loc[labels.loc[:, "pid"].isin(train_data["pid"]), :]

        model = linear_model.Lasso(alpha=0.01, random_state=415)
        model.fit(train_data.iloc[:, 2:], labels[label])

        scores = cross_validate(model, train_data.iloc[:, 2:], labels[label], cv=cv, scoring=regression_scoring)
        print("Model for", label, "scored on 10-fold CV with:")
        for key in regression_scoring.keys():
            print(key, "=", scores["test_" + key])
        print()

        path_to_test = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/LABEL_Heartrate_test_imputed_v.0.0.47.csv'
        test_data = pandas.read_csv(path_to_test)

        predictions = model.predict(test_data.iloc[:, 2:])

        predictions = pandas.DataFrame(predictions)
        predictions.insert(0, 'pid', test_data['pid'].values)

    else:
        raise ValueError("Unknown label!")

    return predictions


if __name__ == "__main__":

    """ Classification setting for subtasks 1, 2 
    
    # processes = []
    # start_time = time.time()
    # 
    # imputations = ["impute_iter_const", "impute_iter_mean", "impute_iter_mean_ids", "impute_iter_most_freq",
    #                "impute_simple_const", "impute_simple_const_ids", "impute_simple_most_freq"]
    # 
    # for i in range(len(imputations)):
    #     for j in range(len(subtask_1_labels)):
    #         p = multiprocessing.Process(target=run_label_specific_svm, args=(label,imputation))
    #         processes.append(p)
    #         p.start()
    # 
    # for process in processes:
    #     process.join()
    #     
    # print('All done within', int((time.time() - start_time) // 3600 + 1), "hours")
    
    """

    """ Regression setting for subtask 3
    
    # start_time = time.time()
    # 
    # imputations = ["impute_iter_const", "impute_iter_mean", "impute_iter_mean_ids", "impute_iter_most_freq",
    #                "impute_simple_const", "impute_simple_const_ids", "impute_simple_most_freq"]
    # 
    # for i in trange(len(imputations)):
    #     for j in trange(len(subtask_3_labels)):
    #         run_label_specific_regression(['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2'][j], imputations[i])
    # 
    # print('All done within', int((time.time() - start_time) // 3600 + 1), "hours")
    
    """


    """ Predictions on test """

    start_time = time.time()

    path_to_save_to = "/Users/andreidm/ETH/courses/iml-tasks/project_2/res/test/"

    # the_rest_of_subtask_1= ['LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

    all_labels = [*subtask_1_labels, *subtask_2_labels]

    for j in trange(len(all_labels)):

        result = train_model_and_predict_on_test(all_labels[j])
        result.to_csv(path_to_save_to + "predictions_" + all_labels[j] + "_" + version + ".csv")
        print("results for", all_labels[j], "saved")

    print('All done within', int((time.time() - start_time) // 3600 + 1), "hours")