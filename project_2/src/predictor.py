
import pandas, numpy, time
from project_2.src import preprocessing
from constants import train_path, train_labels_path
from constants import subtask_1_labels
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def create_svm_models(C_range, random_seed):
    """ This method initialises and returns SVM models with parameters. """

    svm_models = []
    for C in C_range:
        model = ("svm_sigmoid_" + str(C), SVC(C=C, kernel="sigmoid", random_state=random_seed))
        svm_models.append(model)

    return svm_models


if __name__ == "__main__":

    full_data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    data_with_nans = full_data.iloc[:,3:]

    random_seeds = [42, 111, 666, 121, 321, 7, 26, 33, 222, 842]

    start_time = time.time()

    features = preprocessing.get_engineered_features(numpy.array(data_with_nans))  # slow
    timepoint_1 = time.time()
    print(timepoint_1 - start_time, "s for feature engineering")

    imputed_features = preprocessing.impute_data_with_strategies(features)
    timepoint_2 = time.time()
    print(timepoint_2 - timepoint_1, "s for imputation\n")

    for i in range(len(imputed_features)):

        timepoint_1 = time.time()
        print("working with:", imputed_features[i][0])

        imputed_scaled_features = preprocessing.scale_data_with_methods(imputed_features[i][1])
        timepoint_2 = time.time()
        print(timepoint_2 - timepoint_1, "s for scaling\n")

        for j in range(len(imputed_scaled_features)):
            for n_folds in range(5, 20):
                for r in range(len(random_seeds)):

                    svm_models = create_svm_models([10 ** x for x in range(-6, 7)], random_seeds[r])
                    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seeds[r])

                    # TODO: create a convenient data structure to collect results

                    results = []
                    names = []

                    # iterate over models
                    for name, model in svm_models:
                        # iterate over labels to predict
                        for label in subtask_1_labels:

                            # TODO: figure out how sigmoid results will be compared to binary labels
                            result = cross_val_score(model, X, labels.loc[label], cv=cv)

                            names.append(name)
                            results.append(result)



            pass



            print()


    pass