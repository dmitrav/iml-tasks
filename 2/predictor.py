
import pandas, numpy
from .constants import train_path, train_labels_path, test_path

if __name__ == "__main__":

    data = pandas.read_csv(train_path)
    labels = pandas.read_csv(train_labels_path)

    pass