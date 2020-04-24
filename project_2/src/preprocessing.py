import numpy, pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from project_2.src.constants import train_path, test_path, train_labels_path, version


def scale_data_with_methods(imputed_data):
    """ This method scales imputed data accroding to several methods. """

    scaled_data = [
        ("none", imputed_data),
        # ("standard_scaler", StandardScaler().fit_transform(imputed_data)),
        ("min_max_scaler", MinMaxScaler().fit_transform(imputed_data)),
        # ("max_abs_scaler", MaxAbsScaler().fit_transform(imputed_data)),
        ("robust_scaler", RobustScaler(quantile_range=(25, 75)).fit_transform(imputed_data)),
        # ("power_yj_scaler", PowerTransformer(method='yeo-johnson').fit_transform(imputed_data)),
        ("quantile_norm_scaler", QuantileTransformer(output_distribution='normal').fit_transform(imputed_data)),
        # ("quantile_uni_scaler", QuantileTransformer(output_distribution='uniform').fit_transform(imputed_data)),
        ("l2_scaler", Normalizer().fit_transform(imputed_data))  # sample-wise L2
    ]

    return scaled_data


def impute_data_with_strategies(data, random_seed=555):
    """ This method makes imputations to the data (with specified random seed). """

    imputed_data = [
        ("impute_simple_mean", SimpleImputer(strategy="mean").fit_transform(data)),
        ("impute_simple_median", SimpleImputer(strategy="median").fit_transform(data)),
        ("impute_simple_const", SimpleImputer(strategy="constant").fit_transform(data)),
        ("impute_simple_most_freq", SimpleImputer(strategy="most_frequent").fit_transform(data)),
        ("impute_iter_mean", IterativeImputer(initial_strategy="mean", random_state=random_seed).fit_transform(data))
        # ("impute_iter_median", IterativeImputer(initial_strategy="median", random_state=random_seed).fit_transform(data)),
        # ("impute_iter_const", IterativeImputer(initial_strategy="constant", random_state=random_seed).fit_transform(data)),
        # ("impute_iter_most_freq", IterativeImputer(initial_strategy="most_frequent", random_state=random_seed).fit_transform(data))
    ]

    return imputed_data


def get_engineered_features(dataset):
    """ This method gets dataset with time series variables (12 values) for each patient,
        and transforms it to several single valued variables for each patient. """

    # create an empty array for every feature of every patient
    new_dataset = [[[] for x in range(dataset.shape[1])] for x in range(dataset.shape[0] // 12)]

    for j in range(dataset.shape[1]):
        for i in range(dataset.shape[0] // 12):

            patient_records = dataset[i*12:(i+1)*12, j]
            finite_records = patient_records[numpy.isfinite(patient_records)]  # remove nans

            if finite_records.shape[0] > 0:
                # use only finite records if there are any, otherwise only nans will be produced
                patient_records = finite_records

            if finite_records.shape[0] > 1:
                # if there are at least 2 points, get the slope
                linear_slope = numpy.polyfit([x for x in range(1,patient_records.shape[0]+1)], patient_records, 1)[0]
            else:
                linear_slope = 0

            # collect new features
            new_features = [
                numpy.median(patient_records),
                numpy.min(patient_records),
                numpy.max(patient_records),
                numpy.var(patient_records),
                numpy.sum(numpy.isfinite(patient_records)),  # number of finite values
                numpy.sum(patient_records),  # "auc"
                linear_slope
            ]

            new_dataset[i][j].extend(new_features)

    # reshape data structure to a matrix
    flattened = []
    for i in range(len(new_dataset)):
        patient_features = []
        for j in range(len(new_dataset[i])):
            patient_features.extend(new_dataset[i][j])
        flattened.append(patient_features)

    return numpy.array(flattened)


def engineer_and_save_features():
    """ Obvious. """

    full_data = pandas.read_csv(train_path)

    data_with_nans = full_data.iloc[:, 3:]

    features = get_engineered_features(numpy.array(data_with_nans))

    path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/"
    pandas.DataFrame(features).to_csv(path + "engineered_features_" + version + ".csv")


def impute_features_with_strategies_and_save(path):
    """ Impute features and save datasets. """

    features = pandas.read_csv(path)

    imputed_features = impute_data_with_strategies(numpy.array(features))

    save_to_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/"

    for name, data in imputed_features:
        pandas.DataFrame(data).to_csv(save_to_path + name + "_" + version + ".csv")


if __name__ == "__main__":

    features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/engineered_features_v.0.0.12.csv"
    features = numpy.array(pandas.read_csv(features_path)[:,1:])

    print()

