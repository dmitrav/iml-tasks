import numpy, pandas, time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from imblearn.combine import SMOTETomek
from project_2.src.constants import train_path, test_path, train_labels_path, version
from project_2.src.constants import subtask_1_labels, subtask_3_labels, subtask_2_labels
from project_2.src import data_analysis


def scale_data_with_methods(imputed_data):
    """ This method scales imputed data accroding to several methods. """

    scaled_data = [
        # ("not_scaled", imputed_data),
        ("standard_scaler", StandardScaler().fit_transform(imputed_data)),
        # ("max_abs_scaler", MaxAbsScaler().fit_transform(imputed_data)),
        ("power_yj_scaler", PowerTransformer(method='yeo-johnson').fit_transform(imputed_data)),
        # ("quantile_norm_scaler", QuantileTransformer(output_distribution='normal').fit_transform(imputed_data))
        # ("quantile_uni_scaler", QuantileTransformer(output_distribution='uniform').fit_transform(imputed_data))
        # ("l2_scaler", Normalizer().fit_transform(imputed_data)),  # sample-wise L2
        # ("robust_scaler", RobustScaler(quantile_range=(25, 75)).fit_transform(imputed_data)),
        # ("min_max_scaler", MinMaxScaler().fit_transform(imputed_data))
    ]

    return scaled_data


def impute_data_with_strategies(data, random_seed=777):
    """ This method makes imputations to the data (with specified random seed). """

    imputed_data = [
        # ("impute_simple_mean", SimpleImputer(strategy="mean").fit_transform(data)),
        # ("impute_simple_median", SimpleImputer(strategy="median").fit_transform(data)),
        ("impute_simple_const", SimpleImputer(strategy="constant").fit_transform(data)),
        # ("impute_simple_const", SimpleImputer(strategy="constant", add_indicator=True).fit_transform(data)),
        ("impute_simple_most_freq", SimpleImputer(strategy="most_frequent").fit_transform(data)),
        ("impute_iter_mean", IterativeImputer(initial_strategy="mean", random_state=random_seed).fit_transform(data)),
        ("impute_iter_mean", IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True).fit_transform(data)),
        # ("impute_iter_median", IterativeImputer(initial_strategy="median", random_state=random_seed, add_indicator=True).fit_transform(data)),
        ("impute_iter_const", IterativeImputer(initial_strategy="constant", random_state=random_seed).fit_transform(data)),
        ("impute_iter_most_freq", IterativeImputer(initial_strategy="most_frequent", random_state=random_seed).fit_transform(data))
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


def flatten_and_save_features():
    """ This method takes 12 time-series values for each patient and each feature
        and makes 12 features with single values (flattens the time-series).

        SLOW, but works. """

    full_data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    data_with_nans = full_data.iloc[:, 3:]

    new_dataset = [[[] for x in range(data_with_nans.shape[1])] for x in range(data_with_nans.shape[0] // 12)]

    for j in range(data_with_nans.shape[1]):
        for i in range(data_with_nans.shape[0] // 12):
            patient_records = data_with_nans.iloc[i * 12:(i + 1) * 12, j]
            new_dataset[i][j].extend(patient_records.tolist())

    # reshape data structure to a matrix
    flattened = []
    for i in range(len(new_dataset)):
        patient_features = []
        for j in range(len(new_dataset[i])):
            patient_features.extend(new_dataset[i][j])
        flattened.append(patient_features)

    flattened_data = pandas.DataFrame(flattened)

    assert flattened_data.shape[0] == labels.shape[0]
    # since the order in features was not changed, assign ids to features
    flattened_data.insert(0, 'pid', labels['pid'])

    path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/"
    flattened_data.to_csv(path + "flattened_features_" + version + ".csv")


def engineer_and_save_features():
    """ This method call feature engineering routine and saves results. """

    full_data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    data_with_nans = full_data.iloc[:, 3:]

    features = get_engineered_features(numpy.array(data_with_nans))
    features = pandas.DataFrame(features)

    assert features.shape[0] == labels.shape[0]
    # since the order in features was not changed, assign ids to features
    features.insert(0, 'pid', labels['pid'])

    path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/"
    features.to_csv(path + "engineered_features_" + version + ".csv")


def impute_features_with_strategies_and_save(path):
    """ Impute features and save datasets. """

    features = pandas.read_csv(path).iloc[:,1:]

    imputed_features = impute_data_with_strategies(numpy.array(features.iloc[:,1:]))

    if path.split("/")[-1].startswith("LABEL"):
        same_folder = "/".join(path.split("/")[0:-1]) + "/" + path.split("/")[-1].split("_")[0] + "_" + path.split("/")[-1].split("_")[1] + "_"
        if path.split("/")[-1].split("_")[2] == "direct" or path.split("/")[-1].split("_")[2] == "total":
            same_folder += path.split("/")[-1].split("_")[2] + "_"  # fucking bilirubin
    else:
        same_folder = "/".join(path.split("/")[0:-1]) + "/"

    for name, data in imputed_features:
        data = pandas.DataFrame(data)
        data.insert(0, 'pid', features['pid'])
        data.to_csv(same_folder + name + "_" + version + ".csv")


def generate_label_specific_features(features, labels):
    """ This method takes engineered features with nans as input,
        performs filtering of patients with high percent of nans for each label separately
        (to decrease imbalance in data), and saves the resulting datasets. """

    # check how imbalanced labels are initially
    initial_positive_class_percent = numpy.sum(labels.loc[:, subtask_2_labels], 0) / labels.shape[0] * 100

    positive_class_percent = []

    for label in subtask_2_labels:

        # get pid of patients that are of negative and positive classes
        negative_class_pid = labels.loc[labels.loc[:, label] == 0, "pid"]
        positive_class_pid = labels.loc[labels.loc[:, label] == 1, "pid"]

        print(label, ", initial size of positive class: ", positive_class_pid.shape[0], sep="")

        # get corresponding features of negative class
        negative_class_features = features.loc[features.loc[:, "pid"].isin(negative_class_pid), :]

        # among those, get pid of patients that have >= certain % of nans
        if initial_positive_class_percent[label] < 20:
            percent = 0.8  # for Sepsis -> results in 25% of the positive class
            # percent = 0.75  # for engineered features
            # percent = 0.28  # for flattened features
        else:
            percent = 0.5  # for engineered features
            # percent = 0.15  # for flattened features

        low_percent_finite_values_pid = negative_class_features.loc[numpy.sum(numpy.isfinite(negative_class_features.iloc[:, 1:]), 1) / negative_class_features.shape[1] >= percent, "pid"]

        filtered_pid = numpy.concatenate((positive_class_pid, low_percent_finite_values_pid), axis=None)

        # subset those from the initial dataset
        new_features = features.loc[features.loc[:, "pid"].isin(filtered_pid), :]
        new_labels = labels.loc[labels.loc[:, "pid"].isin(filtered_pid), :]

        print(label, ", new features shape: ", new_features.shape[0], sep="")

        # check how imbalanced labels are now
        positive_class_percent.append(numpy.sum(new_labels.loc[:, label], 0) / new_labels.shape[0] * 100)

        path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/label_specific/"
        new_features.to_csv(path + label + "_features_" + version + ".csv")

        print(label, ": dataset saved\n", sep="")

    print("\npositive class imbalance:")
    print("before:", initial_positive_class_percent.tolist())
    print("after:", positive_class_percent)


if __name__ == "__main__":

    """ STEP 0: engineer features """
    pass

    """ STEP 1: generate label-specific features by down-sampling over-represented negative class """

    # features_path = "/Users/dmitrav/ETH/courses/iml-tasks/project_2/data/engineered_features_v.0.0.14.csv"
    # features = pandas.read_csv(features_path)
    # labels = pandas.read_csv(train_labels_path)
    #
    # # take engineered features with nans and
    # generate_label_specific_features(features, labels)

    """ STEP 2: impute label-specific features with different strategies """

    folder = "/Users/dmitrav/ETH/courses/iml-tasks/project_2/data/label_specific/"
    ending = "_features_v.0.0.28.csv"

    # impute them
    for label in subtask_2_labels:
        path = folder + label + ending
        print("imputing ", label, "...", sep="")
        impute_features_with_strategies_and_save(path)
        print("saved\n")



