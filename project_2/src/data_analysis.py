
import pandas, numpy, time
from project_2.src.constants import train_path, test_path, train_labels_path
from project_2.src.constants import subtask_1_labels
from matplotlib import pyplot


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
        data = full_data.dropna(thresh=i + 1)  # drop row if it has < i+project_1 finite values
        x.append(i + 1)
        y.append(data.shape[0])

    # plot size of the dataset over minimal number of finite values in each row
    pyplot.plot(x, y)
    pyplot.xticks(range(full_data.shape[1]))
    pyplot.grid()
    pyplot.show()


if __name__ == "__main__":

    full_data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    pandas.set_option('display.max_columns', 50)
    print(full_data.iloc[:, 3:].describe())