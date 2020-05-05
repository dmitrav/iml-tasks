
import pandas, numpy, time, json
from project_2.src.constants import train_path, test_path, train_labels_path
from project_2.src.constants import subtask_1_labels, subtask_2_labels, subtask_3_labels, version
from matplotlib import pyplot
import seaborn


def inspect_number_of_finite_values():
    """ Naive hypothesis was that number of nans and finite values could be indicative of response variable:
        - by visual inspection of resulting dfs, it doesn't seem right for any.

        VERY SLOW in debug, A LOT OF MEMORY. """

    full_data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    for label in subtask_1_labels:

        start_time = time.time()

        feature = label.replace("LABEL_", "")
        data_slice = full_data.loc[numpy.isfinite(full_data.loc[:, feature]), ['pid', feature]]  # remove nans

        table = []

        for pid in labels['pid']:
            # cound number of finite values
            number_of_values = int(data_slice[data_slice['pid'] == pid].shape[0])
            response = float(labels.loc[labels['pid'] == pid, label])  # get the label
            table.append([pid, number_of_values, response])

        result = pandas.DataFrame(table)
        result.columns = ["pid", "number_of_values", label]

        print(time.time() - start_time, "s elapsed for iteration")


def inspect_nan_values():
    """ See, how many nan values there are in the datasets:
        - every row has nans,
        - after 8 finite values in each row there's a big drop in data size
        """

    full_data = pandas.read_csv(train_path)

    full_data.dropna()  # removes the entire dataset, since each row has at least one nan
    full_data.dropna(axis=1)  # removes all columns with features

    x = [0]
    y = [full_data.shape[0]]
    for i in range(full_data.shape[1]):
        data = full_data.dropna(thresh=i + 1)  # drop row if it has < i+1 finite values
        x.append(i + 1)
        y.append(data.shape[0])

    # plot size of the dataset over minimal number of finite values in each row
    pyplot.plot(x, y)
    pyplot.xticks(range(full_data.shape[1]))
    pyplot.grid()
    pyplot.show()


def plot_distributions_of_finite_values_percent(features=None):
    """ This method plots distributions of finite values in engineered feature matrix:
        - per feature,
        - per sample. """

    if features is None:
        features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/engineered_features_v.0.0.13.csv"
        features = numpy.array(pandas.read_csv(features_path))

    finite_values_percent_per_sample = numpy.sum(numpy.isfinite(features[:, 1:]), 1) / features.shape[1]
    finite_values_percent_per_feature = numpy.sum(numpy.isfinite(features[:, 1:]), 0) / features.shape[0]

    pyplot.figure(figsize=(10, 7), dpi=80)
    kwargs = dict(hist_kws={'alpha': .5}, kde_kws={'linewidth': 2})
    seaborn.distplot(finite_values_percent_per_feature, color="orange", label="per feature", **kwargs)
    seaborn.distplot(finite_values_percent_per_sample, color="dodgerblue", label="per sample", **kwargs)
    pyplot.legend()
    pyplot.show()


def check_imbalance_of_labels():
    """ Trivial method to assess how imbalanced classes / labels are. """

    labels = pandas.read_csv(train_labels_path)

    # check how imbalanced labels are
    positive_class_percent = numpy.sum(labels.loc[:, subtask_2_labels], 0) / labels.shape[0] * 100
    print(positive_class_percent)


def plot_distributions_of_labels():
    """ This is for subtask 3. """

    labels = pandas.read_csv(train_labels_path)

    f, axes = pyplot.subplots(2, 2, figsize=(10, 7), dpi=80)
    kwargs = dict(hist_kws={'alpha': .5}, kde_kws={'linewidth': 2})

    seaborn.distplot(labels.loc[:, subtask_3_labels[0]], label=subtask_3_labels[0], **kwargs, ax=axes[0,0])
    seaborn.distplot(labels.loc[:, subtask_3_labels[1]], label=subtask_3_labels[1], **kwargs, ax=axes[0,1])
    seaborn.distplot(labels.loc[:, subtask_3_labels[2]], label=subtask_3_labels[2], **kwargs, ax=axes[1,0])
    seaborn.distplot(labels.loc[:, subtask_3_labels[3]], label=subtask_3_labels[3], **kwargs, ax=axes[1,1])

    pyplot.legend()
    pyplot.show()


def find_best_models_from_run():
    """ This method sorts methods for each label by sum of all metrics,
        and prints scores for 10 best models for each label. """

    folder = "/Users/andreidm/ETH/courses/iml-tasks/project_2/res/"

    for label in ['LABEL_Lactate',  'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_SaO2']:
    # for label in subtask_1_labels:

        results_file = "results_flattened_" + label + "_impute_iter_mean_v.0.0.27.json"
        with open(folder + results_file, 'r') as file:
            results = json.load(file)

        run_scores_sums = []
        for run in results["svm"]:
            run_sum = 0
            for metric_name in run["scores"].keys():
                run_sum += sum(run["scores"][metric_name])
            run_scores_sums.append(run_sum)

        best_scores = sorted(run_scores_sums, reverse=True)[0:10]
        indices = [run_scores_sums.index(score) for score in best_scores]

        print("Best scores for ", label, ":", sep="")
        print()
        for index in indices:
            full_model_description = results["svm"][index]['model'] + " + " + results["svm"][index][
                'imputation'] + " + " + results["svm"][index]['scaling']
            print("Model ", index + 1, ": ", full_model_description, sep="")
            print('\taccuracy:', numpy.mean(results["svm"][index]["scores"]['accuracy']))
            print('\tprecision:', numpy.mean(results["svm"][index]["scores"]['precision']))
            print('\trecall:', numpy.mean(results["svm"][index]["scores"]['recall']))
            print('\troc_auc:', numpy.mean(results["svm"][index]["scores"]['roc_auc']))
            print('\tf1:', numpy.mean(results["svm"][index]["scores"]['f1']))
            print()


def find_best_regression_models_from_run():
    """ This method sorts methods for each label by sum of all metrics,
        and prints scores for 10 best models for each label. """

    folder = "/Users/dmitrav/ETH/courses/iml-tasks/project_2/res/subtask_3/run_1/"

    for label in subtask_3_labels:

        results_file = "results_" + label + "_impute_iter_const_v.0.0.35.json"
        with open(folder + results_file, 'r') as file:
            results = json.load(file)

        run_scores_sums = []
        for run in results["results"]:
            run_sum = 0
            for metric_name in run["scores"].keys():
                run_sum += sum(run["scores"][metric_name])
            run_scores_sums.append(run_sum)

        best_scores = sorted(run_scores_sums, reverse=True)[0:10]
        indices = [run_scores_sums.index(score) for score in best_scores]

        print("Best scores for ", label, ":", sep="")
        print()
        for index in indices:
            full_model_description = results["results"][index]['model'] + " + " + results["results"][index]['imputation'] + " + " + results["results"][index]['scaling']
            print("Model ", index + 1, ": ", full_model_description, sep="")
            print('\tmax_error:', numpy.mean(results["results"][index]["scores"]['max_error']))
            print('\tneg_mean_squared:', numpy.mean(results["results"][index]["scores"]['neg_mean_squared']))
            print('\tneg_median_absolute:', numpy.mean(results["results"][index]["scores"]['neg_median_absolute']))
            print('\tr2:', numpy.mean(results["results"][index]["scores"]['r2']))
            print()


if __name__ == "__main__":

    # features_path = "/Users/andreidm/ETH/courses/iml-tasks/project_2/data/flattened_features_v.0.0.26.csv"
    # features = numpy.array(pandas.read_csv(features_path))
    # plot_distributions_of_finite_values_percent(features)

    # find_best_models_from_run()

    # check_imbalance_of_labels()

    # plot_distributions_of_labels()

    find_best_regression_models_from_run()