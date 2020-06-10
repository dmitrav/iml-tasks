
import PIL
import tensorflow as tf
import numpy, pandas, time
from scipy.spatial.distance import pdist
from project_4.src import distance_based
from project_4.src import constants
from tqdm import tqdm
from sklearn.model_selection import train_test_split

train_features = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_1.csv")

features = train_features.iloc[:, 2:]
classes = train_features['class']

# test_features = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data.csv")
# split
X_train, X_val, y_train, y_val = train_test_split(features, classes, stratify=classes, random_state=constants.SEED)

# XGBOOST
distance_based.train_xgb(X_train, y_train, X_val, y_val)
