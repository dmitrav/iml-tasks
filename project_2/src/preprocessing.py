import numpy, pandas, time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from imblearn.combine import SMOTETomek
from project_2.src.constants import train_path, test_path, train_labels_path, version
from project_2.src.constants import subtask_1_labels, subtask_3_labels, subtask_2_labels
from project_2.src import data_analysis
from sklearn.kernel_approximation import Nystroem
from tqdm.auto import trange


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
        ("min_max_scaler", MinMaxScaler().fit_transform(imputed_data))
        # ("nystroem", Nystroem(gamma=.2, random_state=1, n_components=imputed_data.shape[0]).fit_transform(imputed_data))
    ]

    return scaled_data


def impute_data_with_strategies(data, random_seed=777):
    """ This method makes imputations to the data (with specified random seed). """

    imputed_data = [
        # ("impute_simple_mean", SimpleImputer(strategy="mean").fit_transform(data)),
        # ("impute_simple_median", SimpleImputer(strategy="median").fit_transform(data)),
        ("impute_simple_const", SimpleImputer(strategy="constant").fit_transform(data)),
        ("impute_simple_const_ids", SimpleImputer(strategy="constant", add_indicator=True).fit_transform(data)),
        ("impute_simple_most_freq", SimpleImputer(strategy="most_frequent").fit_transform(data)),
        ("impute_iter_mean", IterativeImputer(initial_strategy="mean", random_state=random_seed).fit_transform(data)),
        ("impute_iter_mean_ids", IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True).fit_transform(data)),
        # ("impute_iter_median_ids", IterativeImputer(initial_strategy="median", random_state=random_seed, add_indicator=True).fit_transform(data)),
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


def flatten_and_save_features(train=True):
    """ This method takes 12 time-series values for each patient and each feature
        and makes 12 features with single values (flattens the time-series).

        SLOW, but works. """

    if train:
        full_data = pandas.read_csv(train_path)
        labels = pandas.read_csv(train_labels_path)
        addition = ""
    else:
        full_data = pandas.read_csv(test_path)
        labels = pandas.read_csv(train_labels_path)
        addition = "test/"

    data_with_nans = numpy.array(full_data.iloc[:, 3:])

    new_dataset = [[[] for x in range(data_with_nans.shape[1])] for x in range(data_with_nans.shape[0] // 12)]

    for j in range(data_with_nans.shape[1]):
        for i in range(data_with_nans.shape[0] // 12):
            patient_records = data_with_nans[i * 12:(i + 1) * 12, j]
            new_dataset[i][j].extend(patient_records.tolist())

    # reshape data structure to a matrix
    flattened = []
    for i in range(len(new_dataset)):
        patient_features = []
        for j in range(len(new_dataset[i])):
            patient_features.extend(new_dataset[i][j])
        flattened.append(patient_features)

    flattened_data = pandas.DataFrame(flattened)

    if train:
        assert flattened_data.shape[0] == labels.shape[0]
        # since the order in features was not changed, assign ids to features
        flattened_data.insert(0, 'pid', labels['pid'])

    path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/" + addition
    flattened_data.to_csv(path + "flattened_features_" + version + ".csv")


def engineer_and_save_features(train=True):
    """ This method call feature engineering routine and saves results. """

    if train:
        full_data = pandas.read_csv(train_path)
        labels = pandas.read_csv(train_labels_path)
        addition = ""
    else:
        full_data = pandas.read_csv(test_path)
        addition = "test/"

    data_with_nans = full_data.iloc[:, 3:]

    features = get_engineered_features(numpy.array(data_with_nans))
    features = pandas.DataFrame(features)

    if train:
        assert features.shape[0] == labels.shape[0]
        # since the order in features was not changed, assign ids to features
        features.insert(0, 'pid', labels['pid'])

    path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/" + addition
    features.to_csv(path + "engineered_features_" + version + ".csv")


def engineer_train_and_test_features_and_save():
    """ This method does feature engineering on a merged dataset,
        to guarantee the same dimensionality of the train and test data. """

    train_data = pandas.read_csv(train_path)
    test_data = pandas.read_csv(test_path)

    train_labels = pandas.read_csv(train_labels_path)
    train_shape = train_data.shape[0] // 12

    test_labels = test_data['pid'].unique()

    # stack train and test
    full_data = pandas.concat([train_data, test_data], sort=False)

    data_with_nans = full_data.iloc[:, 2:]

    all_features = get_engineered_features(numpy.array(data_with_nans))
    all_features = pandas.DataFrame(all_features)

    train_features = all_features.iloc[:train_shape, :]
    test_features = all_features.iloc[train_shape:, :]

    assert train_features.shape[0] == train_labels.shape[0]
    assert test_features.shape[0] == test_labels.shape[0]
    # since the order in features was not changed, assign ids to features
    train_features.insert(0, 'pid', train_labels['pid'])
    test_features.insert(0, 'pid', test_labels)

    path_to_save_train = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/"
    path_to_save_test = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/"

    train_features.to_csv(path_to_save_train + "engineered_train_" + version + ".csv")
    test_features.to_csv(path_to_save_test + "engineered_test_" + version + ".csv")


def flatten_test_and_train_features_and_save():
    """ This method does flattening on a merged dataset,
        to guarantee the same dimensionality of the train and test data. """

    train_data = pandas.read_csv(train_path)
    test_data = pandas.read_csv(test_path)

    train_labels = pandas.read_csv(train_labels_path)
    train_shape = train_data.shape[0] // 12

    test_labels = test_data['pid'].values[::12]

    full_data = pandas.concat([train_data, test_data], sort=False)

    data_with_nans = numpy.array(full_data.iloc[:, 2:])

    new_dataset = [[[] for x in range(data_with_nans.shape[1])] for x in range(data_with_nans.shape[0] // 12)]

    for j in range(data_with_nans.shape[1]):
        for i in range(data_with_nans.shape[0] // 12):
            patient_records = data_with_nans[i * 12:(i + 1) * 12, j]
            new_dataset[i][j].extend(patient_records.tolist())

    # reshape data structure to a matrix
    flattened = []
    for i in range(len(new_dataset)):
        patient_features = []
        for j in range(len(new_dataset[i])):
            patient_features.extend(new_dataset[i][j])
        flattened.append(patient_features)

    flattened_data = pandas.DataFrame(flattened)

    flattened_train = flattened_data.iloc[:train_shape, :]
    flattened_test = flattened_data.iloc[train_shape:, :]

    assert flattened_train.shape[0] == train_labels.shape[0]
    assert flattened_test.shape[0] == test_labels.shape[0]
    # since the order in features was not changed, assign ids to features
    flattened_train.insert(0, 'pid', train_labels['pid'])
    flattened_test.insert(0, 'pid', test_labels)

    path_to_save_train = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/"
    path_to_save_test = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/"

    flattened_train.to_csv(path_to_save_train + "flattened_train_" + version + ".csv")
    flattened_test.to_csv(path_to_save_test + "flattened_test_" + version + ".csv")


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


def generate_label_specific_features_for_regression(features, labels):
    """ This is for subtask 3. Some filtering is done for each label. """

    # should habe ENGINEERED features as input
    labels_to_process = ["LABEL_ABPm"]

    # should habe FLATTENED features as input
    labels_to_process = ["LABEL_RRate", 'LABEL_SpO2', 'LABEL_Heartrate']

    for label in labels_to_process:

        # hardcoded interval of normal values, based on distribution plots
        if "RRate" in label:
            interval = (10, 35)
        elif "ABPm" in label:
            interval = (50, 130)
        elif "SpO2" in label:
            interval = (90, 100)
        elif "Heartrate" in label:
            interval = (40, 140)
        else:
            raise ValueError("Hardcoded labels are wrong.")

        # get pids of normal values
        normal_values_indices = labels.loc[(interval[0] < labels.loc[:, label]) & (labels.loc[:, label] < interval[1]), "pid"]
        # filter out the rest
        normal_features = features.loc[features.loc[:, "pid"].isin(normal_values_indices), :]

        print("\noutliers removal for", label)
        print("size before:", features.shape[0])
        print("size after:", normal_features.shape[0])

        path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/"
        normal_features.to_csv(path + label + "_flattened_" + version + ".csv")

        print(label, ": dataset saved", sep="")


def generate_label_specific_features(features, labels):
    """ This method takes engineered features with nans as input,
        performs filtering of patients with high percent of nans for each label separately
        (to decrease imbalance in data), and saves the resulting datasets. """

    # SUBTASK 1, should have ENGINEERED features as input
    labels_to_process = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

    # SUBTASK 1, should have FLATTENED features as input
    labels_to_process = ['LABEL_AST', 'LABEL_Alkalinephos']

    # SUBTASK 2, should have FLATTENED features as input
    labels_to_process = ['LABEL_Sepsis']

    # check how imbalanced labels are initially
    initial_positive_class_percent = numpy.sum(labels.loc[:, labels_to_process], 0) / labels.shape[0] * 100

    positive_class_percent = []

    for label in labels_to_process:

        # get pid of patients that are of negative and positive classes
        negative_class_pid = labels.loc[labels.loc[:, label] == 0, "pid"]
        positive_class_pid = labels.loc[labels.loc[:, label] == 1, "pid"]

        print(label, ", initial size of positive class: ", positive_class_pid.shape[0], sep="")

        # get corresponding features of negative class
        negative_class_features = features.loc[features.loc[:, "pid"].isin(negative_class_pid), :]

        # among those, get pid of patients that have >= certain % of nans
        if initial_positive_class_percent[label] < 20:
            percent = 0.27  # Sepsis: flattened features -> results in 25% of the positive class
            # percent = 0.8  # Sepsis: engineered features -> results in 25% of the positive class
            # percent = 0.75  # for engineered features
            # percent = 0.28  # for flattened features
        else:
            # percent = 0.5  # for engineered features
            percent = 0.15  # for flattened features

        low_percent_finite_values_pid = negative_class_features.loc[numpy.sum(numpy.isfinite(negative_class_features.iloc[:, 1:]), 1) / negative_class_features.shape[1] >= percent, "pid"]

        filtered_pid = numpy.concatenate((positive_class_pid, low_percent_finite_values_pid), axis=None)

        # subset those from the initial dataset
        new_features = features.loc[features.loc[:, "pid"].isin(filtered_pid), :]
        new_labels = labels.loc[labels.loc[:, "pid"].isin(filtered_pid), :]

        print(label, ", new features shape: ", new_features.shape[0], sep="")

        # check how imbalanced labels are now
        positive_class_percent.append(numpy.sum(new_labels.loc[:, label], 0) / new_labels.shape[0] * 100)

        path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/"
        new_features.to_csv(path + label + "_flattened_" + version + ".csv")

        print(label, ": dataset saved\n", sep="")

    print("\npositive class imbalance:")
    print("before:", initial_positive_class_percent.tolist())
    print("after:", positive_class_percent)


def impute_and_scale_train_and_test_features_and_save(label, random_seed=777):

    engineered_test_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/engineered_test_v.0.0.45.csv"
    flattened_test_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/flattened_test_v.0.0.45.csv"

    if label == 'LABEL_BaseExcess':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_BaseExcess_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        # combine train and test
        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="most_frequent", random_state=random_seed)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:,1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_Fibrinogen':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Fibrinogen_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = SimpleImputer(strategy="constant")
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = StandardScaler()
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_AST':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_AST_flattened_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(flattened_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = StandardScaler()
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_Alkalinephos':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Alkalinephos_flattened_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(flattened_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_Bilirubin_total':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Bilirubin_total_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="most_frequent", random_state=random_seed)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_Lactate':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Lactate_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="constant", random_state=random_seed)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = StandardScaler()
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_TroponinI':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_TroponinI_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = SimpleImputer(strategy="most_frequent")
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_SaO2':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_SaO2_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="most_frequent", random_state=random_seed)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_Bilirubin_direct':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Bilirubin_direct_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_EtCO2':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_EtCO2_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = StandardScaler()
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    # subtask 2
    elif label == 'LABEL_Sepsis':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Sepsis_flattened_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(flattened_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    # subtask 3
    elif label == 'LABEL_RRate':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_RRate_flattened_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(flattened_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="most_frequent", random_state=random_seed)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = MinMaxScaler()
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_ABPm':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_ABPm_engineered_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(engineered_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = SimpleImputer(strategy="constant", add_indicator=True)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = MinMaxScaler()
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_SpO2':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_SpO2_flattened_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(flattened_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="mean", random_state=random_seed, tol=0.35, add_indicator=True, verbose=2)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = PowerTransformer(method='yeo-johnson')
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    elif label == 'LABEL_Heartrate':

        train_data = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/LABEL_Heartrate_flattened_v.0.0.46.csv")
        train_data = train_data.drop(train_data.columns[0:2], axis=1)
        test_data = pandas.read_csv(flattened_test_path)
        test_data = test_data.drop(test_data.columns[0], axis=1)

        train_shape = train_data.shape[0]

        assert train_data.shape[1] == test_data.shape[1]
        data = pandas.concat([train_data, test_data], sort=False)

        imputer = IterativeImputer(initial_strategy="mean", random_state=random_seed, add_indicator=True)
        # fit imputer only on train data
        imputer.fit(train_data.iloc[:, 1:])
        # impute train and test together
        imputed_data = imputer.transform(data.iloc[:, 1:])

        scaler = MinMaxScaler()
        # fit scaler only on imputed train data
        scaler.fit(imputed_data[:train_shape, :])
        # scale train and test together
        scaled_data = scaler.transform(imputed_data)

        train_features = pandas.DataFrame(scaled_data).iloc[:train_shape, :]
        train_features.insert(0, 'pid', data['pid'].values[:train_shape])

        test_features = pandas.DataFrame(scaled_data).iloc[train_shape:, :]
        test_features.insert(0, 'pid', data['pid'].values[train_shape:])

    else:
        raise ValueError("Unknown label!")

    path_to_save_train = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/"
    path_to_save_test = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/"

    train_features.to_csv(path_to_save_train + label + "_train_imputed_" + version + ".csv")
    test_features.to_csv(path_to_save_test + label + "_test_imputed_" + version + ".csv")


def impute_and_scale_and_save_test_features(label, random_seed=777):
    """ Impute and scale test set for each label:
        works ONLY for add_indicator=False. """

    path = '/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/'

    if label in ['LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Sepsis', 'LABEL_RRate', 'LABEL_SpO2', 'LABEL_Heartrate']:
        ending = 'flattened_features_v.0.0.37.csv'
    else:
        ending = 'engineered_features_v.0.0.37.csv'

    data = pandas.read_csv(path + ending)

    if label == 'LABEL_BaseExcess':
        imputed_data = IterativeImputer(initial_strategy="most_frequent", random_state=random_seed).fit_transform(data)
        scaled_data = PowerTransformer(method='yeo-johnson').fit_transform(imputed_data)

    elif label == 'LABEL_Fibrinogen':
        imputed_data = SimpleImputer(strategy="constant").fit_transform(data)
        scaled_data = StandardScaler().fit_transform(imputed_data)

    elif label == 'LABEL_Bilirubin_total':
        imputed_data = IterativeImputer(initial_strategy="most_frequent", random_state=random_seed).fit_transform(data)
        scaled_data = PowerTransformer(method='yeo-johnson').fit_transform(imputed_data)

    elif label == 'LABEL_Lactate':
        imputed_data = IterativeImputer(initial_strategy="constant", random_state=random_seed).fit_transform(data)
        scaled_data = StandardScaler().fit_transform(imputed_data)

    elif label == 'LABEL_TroponinI':
        imputed_data = SimpleImputer(strategy="most_frequent").fit_transform(data)
        scaled_data = PowerTransformer(method='yeo-johnson').fit_transform(imputed_data)

    # subtask 3
    elif label == 'LABEL_RRate':
        imputed_data = IterativeImputer(initial_strategy="most_frequent", random_state=random_seed).fit_transform(data)
        scaled_data = MinMaxScaler().fit_transform(imputed_data)

    else:
        raise ValueError("Unknown label!")

    scaled_data = pandas.DataFrame(scaled_data)
    scaled_data.to_csv(path + label + "_test_features_" + version + ".csv")


if __name__ == "__main__":

    """ Classification preprocessing pipeline. """
    
    # # STEP 0: engineer features
    # engineer_and_save_features(train=False)
    # flatten_and_save_features(train=False)

    # # STEP 1: generate label-specific features by down-sampling over-represented negative class
    #
    # features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/flattened_features_v.0.0.26.csv"
    # features = pandas.read_csv(features_path)
    # labels = pandas.read_csv(train_labels_path)
    #
    # # take engineered features with nans and
    # generate_label_specific_features(features, labels)
    #
    # STEP 2: impute label-specific features with different strategies
    #
    # folder = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/test/"
    # ending = "_flattened_v.0.0.31.csv"
    #
    # # impute them
    # for label in subtask_1_labels:
    #     path = folder + label + ending
    #     print("imputing ", label, "...", sep="")
    #     impute_features_with_strategies_and_save(path)
    #     print("saved\n")
    #
    #
    # """ Regression setting: subtask 3 """
    #
    # # STEP 1: generate label-specific features by removing outliers
    # features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/flattened_features_v.0.0.26.csv"
    # features = pandas.read_csv(features_path)
    # labels = pandas.read_csv(train_labels_path)
    #
    # generate_label_specific_features_for_regression(features, labels)
    #
    # # STEP 2: impute label-specific features with different strategies
    # folder = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/label_specific/flattened/"
    # ending = "_flattened_v.0.0.37.csv"
    #
    # for i in trange(len(subtask_3_labels)):
    #     path = folder + subtask_3_labels[i] + ending
    #     print("imputing ", subtask_3_labels[i], "...", sep="")
    #     impute_features_with_strategies_and_save(path)
    #     print("saved\n")

    """ Impute and scale features for test sets """

    # for label in [*subtask_1_labels, *subtask_3_labels]:
    #     impute_and_scale_and_save_test_features(label)

    """ All preprocessing workflow for train and test together, to cope with add_indicator option while imputing. """

    # engineer_train_and_test_features_and_save()
    # flatten_test_and_train_features_and_save()

    # STEP 1: generate label-specific features by down-sampling over-represented negative class

    # CLASSIFICATION

    # features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/flattened_train_v.0.0.45.csv"
    # features = pandas.read_csv(features_path)
    # labels = pandas.read_csv(train_labels_path)
    #
    # # take engineered features with nans and
    # generate_label_specific_features(features, labels)

    # REGRESSION

    # features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/train/flattened_train_v.0.0.45.csv"
    # features = pandas.read_csv(features_path)
    # labels = pandas.read_csv(train_labels_path)
    #
    # generate_label_specific_features_for_regression(features, labels)

    # STEP 2: impute label-specific features with different strategies

    # the_rest_of_subtask_1 = ['LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
    the_rest_of_subtask_3 = ['LABEL_SpO2']

    labels_to_impute = [*the_rest_of_subtask_3]

    for i in trange(len(labels_to_impute)):

        print("imputing ", labels_to_impute[i], "...", sep="")
        impute_and_scale_train_and_test_features_and_save(labels_to_impute[i])
        print("saved\n")

