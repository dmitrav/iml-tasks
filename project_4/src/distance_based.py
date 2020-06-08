
import PIL, tqdm
import tensorflow as tf
import numpy, pandas
from project_4.src import constants
from tqdm import tqdm

from xgboost import XGBClassifier
from scipy.spatial.distance import pdist



def get_features_from_pretrained_net(model, image):

    img_data = tf.keras.preprocessing.image.img_to_array(image)
    img_data = numpy.expand_dims(img_data, axis=0)

    img_data = tf.keras.applications.inception_v3.preprocess_input(img_data)

    features_3d = model.predict(img_data)

    features_averaged_1d = features_3d[0].mean(axis=0).mean(axis=0)

    return features_averaged_1d


def generate_train_and_test_datasets():

    # # TRAIN

    with open(constants.TRAIN_TRIPLETS) as file:
        train_triplets = file.readlines()

    train_features = []
    train_ids = []
    train_labels = []

    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    for triplet in tqdm(train_triplets):
        name_a, name_b, name_c = [image_id.replace("\n", "") for image_id in triplet.split(" ")]

        # read images from triplet and resize them
        image_a = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_a + ".jpg")
        image_b = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_b + ".jpg")
        image_c = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_c + ".jpg")

        features_a = get_features_from_pretrained_net(model, image_a)
        features_b = get_features_from_pretrained_net(model, image_b)
        features_c = get_features_from_pretrained_net(model, image_c)

        train_features.append(numpy.concatenate([features_a, features_b, features_c]))
        train_ids.append("_".join([name_a, name_b, name_c]))
        train_labels.append(1)

        train_features.append(numpy.concatenate([features_a, features_c, features_b]))
        train_ids.append("_".join([name_a, name_c, name_b]))
        train_labels.append(0)

    train_data = pandas.DataFrame(train_features)
    train_data.insert(0, "class", train_labels)
    train_data.insert(0, "id", train_ids)

    # # TEST

    train_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data.csv", index=False)
    print("train data saved")

    with open(constants.TEST_TRIPLETS) as file:
        test_triplets = file.readlines()

    test_features = []
    test_ids = []

    for triplet in tqdm(test_triplets):
        name_a, name_b, name_c = [image_id.replace("\n", "") for image_id in triplet.split(" ")]

        # read images from triplet and resize them
        image_a = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_a + ".jpg")
        image_b = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_b + ".jpg")
        image_c = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_c + ".jpg")

        features_a = get_features_from_pretrained_net(model, image_a)
        features_b = get_features_from_pretrained_net(model, image_b)
        features_c = get_features_from_pretrained_net(model, image_c)

        test_features.append(numpy.concatenate([features_a, features_b, features_c]))
        test_ids.append("_".join([name_a, name_b, name_c]))

    test_data = pandas.DataFrame(test_features)
    test_data.insert(0, "id", test_ids)

    test_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data.csv", index=False)
    print("test data saved")


def train_xgb_and_predict(X_train, y_train, X_val, y_val):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectPercentile(score_func=mutual_info_classif)),
        ('classifier', XGBClassifier(random_state=constants.SEED))
    ])

    param_grid = {
        'selector__percentile': [20, 50, 95],

        'classifier__learning_rate': [0.1],
        'classifier__n_estimators': [500],
        'classifier__max_depth': [5],
        'classifier__min_child_weight': [1],
        'classifier__gamma': [0.1],
        'classifier__reg_alpha': [0.01],
        'classifier__subsample': [0.8],
        'classifier__colsample_bytree': [0.8],
        'classifier__objective': ['binary:logistic'],
        'classifier__scale_pos_weight': [1]
    }

    clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

    start = time.time()
    clf.fit(X_train, y_train)
    print("training took", round(time.time() - start) // 60 + 1, 'min')

    val_score = clf.score(X_val, y_val)
    print(label, ', best model val auc: ', val_score, sep="")
    print("best params:", clf.best_params_)

    predictions = clf.predict(test_features.iloc[:, 1:])

    predictions = "\n".join([str(prob) for prob in predictions])
    with open("/Users/andreidm/ETH/courses/iml-tasks/project_4/res/xgboost.txt", 'w') as file:
        file.write(predictions)

    print("xgb predictions saved")

    return predictions


def get_accuracy_score_for_metric(a_features, b_features, c_features, classes, metric_name):

    hits = 0
    for i in range(features.shape[0]):

        a_to_b_distance = pdist(numpy.array([a_features[i], b_features[i]]), metric=metric_name)
        a_to_c_distance = pdist(numpy.array([a_features[i], c_features[i]]), metric=metric_name)

        if a_to_b_distance < a_to_c_distance and int(classes[i]) == 1:
            hits += 1
        elif a_to_b_distance >= a_to_c_distance and int(classes[i]) == 0:
            hits += 1
        else:
            pass

    accuracy = hits / features.shape[0]
    return accuracy


def evaluate_distance_based_accuracy(features, classes):

    features = numpy.array(features)

    image_a_features = features[:, :(features.shape[0] // 3)]
    image_b_features = features[:, (features.shape[0] // 3):(2 * features.shape[0] // 3)]
    image_c_features = features[:, (2*features.shape[0] // 3):]
    
    metrics_to_evaluate = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
                           'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
                           'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                           'sqeuclidean', 'yule']

    metrics_scores = []
    for metric in tqdm(metrics_to_evaluate):
        score = get_accuracy_score_for_metric(image_a_features, image_b_features, image_c_features, classes, metric)
        metrics_scores.append(score)

    result = pandas.DataFrame([[metrics_scores]], columns=metrics_to_evaluate)
    result.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/res/distance_based_scores.csv", index=False)


if __name__ == "__main__":

    # generate_train_and_test_datasets()

    train_features = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data.csv")
    test_features = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data.csv")

    features = train_features.iloc[:, 2:]
    classes = train_features['class']

    # DISTANCE BASED ACCURACY
    evaluate_distance_based_accuracy(features, classes)

    # split
    X_train, X_val, y_train, y_val = train_test_split(features, classes, stratify=classes, random_state=constants.SEED)

    start = time.time()

    # XGBOOST
    predictions = train_xgb_and_predict(X_train, y_train, X_val, y_val)