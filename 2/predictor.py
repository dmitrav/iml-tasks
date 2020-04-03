
import pandas, numpy
from constants import train_path, train_labels_path, test_path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from preprocessing import impute_missing_values_with_number, impute_missing_values_iteratively
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.svm import SVC


if __name__ == "__main__":

    full_data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    data_with_nans = full_data.iloc[:,3:]

    random_seeds = [42, 111, 666, 121, 321, 7, 26, 33, 222, 842]

    imputations = ["mean", "median", "constant", "most_frequent",
                   "iter_mean", "iter_median", "iter_constant", "iter_most_frequent"]

    tranformations = ["none", "standard", "min-max", "max-abs", "robust", "power-YJ",
                      "quantile-norm", "quantile-uni", "sample-wise-L2"]

    simply_imputed_data = [
        SimpleImputer(strategy="mean").fit_transform(data_with_nans),
        SimpleImputer(strategy="median").fit_transform(data_with_nans),
        SimpleImputer(strategy="constant").fit_transform(data_with_nans),
        SimpleImputer(strategy="most_frequent").fit_transform(data_with_nans)
    ]

    for r in range(len(random_seeds)):

        svm_models = [
            ("svm_sigmoid_1e-5", SVC(C=1e-5, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e-4", SVC(C=1e-4, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e-3", SVC(C=1e-3, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e-2", SVC(C=1e-2, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e-1", SVC(C=1e-1, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e-0", SVC(C=1e-0, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e1", SVC(C=1e1, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e2", SVC(C=1e2, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e3", SVC(C=1e3, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e4", SVC(C=1e4, kernel="sigmoid", random_state=random_seeds[r])),
            ("svm_sigmoid_1e5", SVC(C=1e5, kernel="sigmoid", random_state=random_seeds[r]))
        ]

        imputed_data = [
            *simply_imputed_data,
            IterativeImputer(initial_strategy="mean", random_seed=random_seeds[r]),
            IterativeImputer(initial_strategy="median", random_seed=random_seeds[r]),
            IterativeImputer(initial_strategy="constant", random_seed=random_seeds[r]),
            IterativeImputer(initial_strategy="most_frequent", random_seed=random_seeds[r])
        ]

        for n in range(5, 20):

            cv = KFold(n_splits=n, shuffle=True, random_state=random_seeds[r])

            for i in range(len(imputed_data)):

                scaled_data = [
                    data,
                    StandardScaler().fit_transform(imputed_data[i]),
                    MinMaxScaler().fit_transform(imputed_data[i]),
                    MaxAbsScaler().fit_transform(imputed_data[i]),
                    RobustScaler(quantile_range=(25, 75)).fit_transform(imputed_data[i]),
                    PowerTransformer(method='yeo-johnson').fit_transform(imputed_data[i]),
                    QuantileTransformer(output_distribution='normal').fit_transform(imputed_data[i]),
                    QuantileTransformer(output_distribution='uniform').fit_transform(imputed_data[i]),
                    Normalizer().fit_transform(imputed_data[i])
                ]

                for s in range(len(scaled_data)):

                    # TODO: response variable y is missing:
                    #  what shape should it have?

                    # results = []
                    # names = []
                    #
                    # for name, model in svm_models:
                    #
                    #     result = cross_val_score(model, X, train_data["Survived"], cv=cv)
                    #     names.append(name)
                    #     results.append(result)



                    pass



            print()


    pass