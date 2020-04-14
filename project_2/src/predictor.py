
import pandas
from project_2.src import preprocessing
from constants import train_path, train_labels_path
from constants import subtask_1_labels
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.impute import SimpleImputer, IterativeImputer
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

    imputed_data = preprocessing.impute_data_with_strategies(data_with_nans)

    for i in range(len(imputed_data)):
        scaled_data = preprocessing.scale_data_with_methods(imputed_data[i])

        for s in range(len(scaled_data)):
            X = preprocessing.get_engineered_features(scaled_data[i])  # slow

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

                            # TODO: incorporate softmax
                            result = cross_val_score(model, X, labels.loc[label], cv=cv)

                            names.append(name)
                            results.append(result)



            pass



            print()


    pass