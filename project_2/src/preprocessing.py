import numpy, pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


def scale_data_with_methods(imputed_data):
    """ This method scales imputed data accroding to several methods. """

    scaled_data = [
        ("not_scaled", imputed_data),
        ("standard_scaled", StandardScaler().fit_transform(imputed_data)),
        ("min_max_scaled", MinMaxScaler().fit_transform(imputed_data)),
        ("max_abs_scaled", MaxAbsScaler().fit_transform(imputed_data)),
        ("robust_scaled", RobustScaler(quantile_range=(25, 75)).fit_transform(imputed_data)),
        ("power_yj_scaled", PowerTransformer(method='yeo-johnson').fit_transform(imputed_data)),
        ("quantile_norm_scaled", QuantileTransformer(output_distribution='normal').fit_transform(imputed_data)),
        ("quantile_uni_scaled", QuantileTransformer(output_distribution='uniform').fit_transform(imputed_data)),
        ("l2_scaled", Normalizer().fit_transform(imputed_data[i]))  # sample-wise L2
    ]

    return scaled_data


def impute_data_with_strategies(data, random_seed=555):
    """ This method makes imputations to the data (with specified random seed). """

    imputed_data = [
        ("impute_simple_mean", SimpleImputer(strategy="mean").fit_transform(data)),
        ("impute_simple_median", SimpleImputer(strategy="median").fit_transform(data)),
        ("impute_simple_const", SimpleImputer(strategy="constant").fit_transform(data)),
        ("impute_simple_most_freq", SimpleImputer(strategy="most_frequent").fit_transform(data)),
        ("impute_iter_mean", IterativeImputer(initial_strategy="mean", random_state=random_seed).fit_transform(data)),
        ("impute_iter_median", IterativeImputer(initial_strategy="median", random_state=random_seed).fit_transform(data)),
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
                # use only finite records if there are any, otherwise only nans are produced
                patient_records = finite_records

            # collect new features
            new_features = [
                numpy.median(patient_records),
                numpy.min(patient_records),
                numpy.max(patient_records),
                numpy.var(patient_records),
                numpy.unique(patient_records).shape[0]
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


    pass