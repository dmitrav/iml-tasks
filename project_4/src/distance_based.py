
import PIL, tqdm
import tensorflow as tf
import numpy, pandas, time
from project_4.src import constants
from tqdm import tqdm

from xgboost import XGBClassifier
from scipy.spatial.distance import pdist
from sklearn.feature_selection import f_classif

from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression, f_classif, mutual_info_classif, mutual_info_regression


def get_features_from_pretrained_net(model, image):

    img_data = tf.keras.preprocessing.image.img_to_array(image)
    img_data = numpy.expand_dims(img_data, axis=0)

    # img_data = tf.keras.applications.inception_v3.preprocess_input(img_data)
    # img_data = tf.keras.applications.inception_resnet_v2.preprocess_input(img_data)
    img_data = tf.keras.applications.xception.preprocess_input(img_data)
    # img_data = tf.keras.applications.nasnet.preprocess_input(img_data)

    features_3d = model.predict(img_data)

    features_averaged_1d = features_3d[0].mean(axis=0).mean(axis=0)

    return features_averaged_1d


def generate_train_and_test_datasets():

    # model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    # model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
    model = tf.keras.applications.Xception(include_top=False, weights='imagenet')
    # model = tf.keras.applications.NASNetLarge(include_top=False, weights='imagenet')

    # # TRAIN
    # with open(constants.TRAIN_TRIPLETS) as file:
    #     train_triplets = file.readlines()
    #
    # train_features = []
    # train_ids = []
    # train_labels = []
    #
    # i = 1
    #
    # for triplet in tqdm(train_triplets):
    #     name_a, name_b, name_c = [image_id.replace("\n", "") for image_id in triplet.split(" ")]
    #
    #     # read images from triplet and resize them
    #     # resize to 331 x 331 for nasnet...
    #     target_size = (331, 331)
    #
    #     image_a = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_a + ".jpg", target_size=target_size)
    #     image_b = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_b + ".jpg", target_size=target_size)
    #     image_c = tf.keras.preprocessing.image.load_img(constants.PATH_TO_RAW_DATA + name_c + ".jpg", target_size=target_size)
    #
    #     features_a = get_features_from_pretrained_net(model, image_a)
    #     features_b = get_features_from_pretrained_net(model, image_b)
    #     features_c = get_features_from_pretrained_net(model, image_c)
    #
    #     train_features.append(numpy.concatenate([features_a, features_b, features_c]))
    #     train_ids.append("_".join([name_a, name_b, name_c]))
    #     train_labels.append(1)
    #
    #     train_features.append(numpy.concatenate([features_a, features_c, features_b]))
    #     train_ids.append("_".join([name_a, name_c, name_b]))
    #     train_labels.append(0)
    #
    #     if len(train_features) == 10000:
    #
    #         train_data = pandas.DataFrame(train_features)
    #
    #         transform_fn = lambda x: pandas.to_numeric(x, downcast='float')
    #         train_data = train_data[train_data.columns].apply(transform_fn)
    #
    #         train_data.insert(0, "id", train_ids)
    #         train_data.insert(0, "class", train_labels)
    #
    #         # train_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_IRNv2_{}.csv".format(i), index=False)
    #         # train_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_X_{}.csv".format(i), index=False)
    #         train_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_NN_{}.csv".format(i), index=False)
    #         print("train data {} saved".format(i))
    #
    #         train_features = []
    #         train_labels = []
    #         train_ids = []
    #         i += 1
    #
    # train_data = pandas.DataFrame(train_features)
    #
    # transform_fn = lambda x: pandas.to_numeric(x, downcast='float')
    # train_data = train_data[train_data.columns].apply(transform_fn)
    #
    # train_data.insert(0, "id", train_ids)
    # train_data.insert(0, "class", train_labels)
    # train_data = train_data.astype({'class': 'int8'})
    #
    # # train_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_IRNv2_{}.csv".format(i), index=False)
    # # train_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_X_{}.csv".format(i), index=False)
    # train_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_NN_{}.csv".format(i), index=False)
    # print("train data saved")

    # TEST

    with open(constants.TEST_TRIPLETS) as file:
        test_triplets = file.readlines()

    test_features = []
    test_ids = []

    i = 1

    # read images from triplet and resize them
    # resize to 331 x 331 for nasnet...
    # target_size = (331, 331)

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

        if len(test_features) == 10000:

            test_data = pandas.DataFrame(test_features)

            transform_fn = lambda x: pandas.to_numeric(x, downcast='float')
            test_data = test_data[test_data.columns].apply(transform_fn)

            test_data.insert(0, "id", test_ids)

            # test_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_IRNv2_{}.csv".format(i), index=False)
            # test_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_X_{}.csv".format(i), index=False)
            test_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_NN_{}.csv".format(i), index=False)
            print("test data {} saved".format(i))

            test_features = []
            test_ids = []
            i += 1

    # save last chunk of data
    test_data = pandas.DataFrame(test_features)

    transform_fn = lambda x: pandas.to_numeric(x, downcast='float')
    test_data = test_data[test_data.columns].apply(transform_fn)

    test_data.insert(0, "id", test_ids)

    # test_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_IRNv2_{}.csv".format(i), index=False)
    # test_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_NN_{}.csv".format(i), index=False)
    test_data.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_X_{}.csv".format(i), index=False)
    print("test data saved")


def train_xgb(X_train, y_train, X_val, y_val):

    # pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        # ('selector', SelectPercentile(score_func=f_classif)),
        # ('classifier', XGBClassifier(random_state=constants.SEED, verbosity=2))
    # ])

    # param_grid = {
        # 'selector__percentile': [90, 100],
    #
    #     'classifier__learning_rate': [0.05],
    #     'classifier__n_estimators': [2000],
    #     'classifier__max_depth': [6],
    #     'classifier__min_child_weight': [3],
    #     'classifier__gamma': [1],
    #     'classifier__reg_alpha': [0.9],
    #     'classifier__reg_lambda': [0.8],
    #     'classifier__subsample': [1.],
    #     'classifier__colsample_bytree': [0.3],
    #     'classifier__objective': ['binary:logistic'],
    #     'classifier__scale_pos_weight': [1],
    #     'classifier__early_stopping_rounds': [10],
    # }

    model = XGBClassifier(random_state=constants.SEED, verbosity=2)

    param_grid = {

        'learning_rate': [0.01],
        'n_estimators': [10000],
        'max_depth': [6],
        'min_child_weight': [3],
        'gamma': [1],
        'reg_alpha': [0.9],
        'reg_lambda': [0.8],
        'subsample': [1.],
        'colsample_bytree': [0.3],
        'objective': ['binary:logistic'],
        'scale_pos_weight': [1]
    }

    # inception-v3:
    # default pars + 100% -> 0.63 accuracy
    # lr = 0.05, max_depth = 8, 500 estimators -> 0.653
    # lr = 0.1, max_depth = 8, 200 estimators, min_child_weight = 3, reg_alpha = 1 -> 0.66
    # lr = 0.01, colsample_bytree = 0.3, subsample = 1., 100 estimators -> 0.66
    # without scaler, lr = 0.005, 100 estimators -> 0.666 + MUCH FASTER

    clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

    fit_params = {"early_stopping_rounds": 100,
                  "eval_set": [[X_val, y_val]]}

    start = time.time()
    print("training started...")
    clf.fit(X_train, y_train, **fit_params)
    print("training took", round(time.time() - start) // 60 + 1, 'min')

    val_score = clf.score(X_val, y_val)
    print('best model val auc: ', val_score, sep="")
    print("best params:", clf.best_params_)

    return clf


def get_accuracy_score_for_metric(a_features, b_features, c_features, classes, metric_name):

    hits = 0
    for i in range(len(classes)):

        a_to_b_distance = pdist([a_features[i].tolist(), b_features[i].tolist()], metric=metric_name)
        a_to_c_distance = pdist([a_features[i].tolist(), c_features[i].tolist()], metric=metric_name)

        if a_to_b_distance < a_to_c_distance and int(classes[i]) == 1:
            hits += 1
        elif a_to_b_distance >= a_to_c_distance and int(classes[i]) == 0:
            hits += 1
        else:
            pass

    accuracy = hits / len(classes)
    return accuracy


def evaluate_distance_based_accuracy(features, classes):

    features = numpy.array(features)

    image_a_features = features[:, :(features.shape[1] // 3)]
    image_b_features = features[:, (features.shape[1] // 3):(2 * features.shape[1] // 3)]
    image_c_features = features[:, (2*features.shape[1] // 3):]

    metrics_scores = []
    for metric in tqdm(constants.DISTANCE_METRICS):
        score = get_accuracy_score_for_metric(image_a_features, image_b_features, image_c_features, classes, metric)
        metrics_scores.append(score)
        print(metric, "scored:", round(score, 3))

    return metrics_scores


def compute_distances_on_train_set():

    path_to_features = "/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_NN_{}.csv"

    dataset_scores = []
    for i in tqdm(range(1, 13)):
        print("train_data_{}".format(i), "is being processed\n")
        train_features = pandas.read_csv(path_to_features.format(i))

        features = train_features.iloc[:, 2:]
        classes = train_features['class']

        # DISTANCE BASED ACCURACY
        metrics_scores = evaluate_distance_based_accuracy(features, classes)
        dataset_scores.append(metrics_scores)

    result = pandas.DataFrame(dataset_scores, columns=constants.DISTANCE_METRICS)
    result.to_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/res/distance_based_scores_NN.csv", index=False)


def predict_similarity_based_on_distance():

    path_to_features = "/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_X_{}.csv"

    predictions = []

    for i in tqdm(range(1, 7)):

        print("test_data_{}".format(i), "is being processed\n")
        test_features = pandas.read_csv(path_to_features.format(i))

        features = numpy.array(test_features.iloc[:, 1:])

        a_features = features[:, :(features.shape[1] // 3)]
        b_features = features[:, (features.shape[1] // 3):(2 * features.shape[1] // 3)]
        c_features = features[:, (2 * features.shape[1] // 3):]

        for i in range(features.shape[0]):

            a_to_b_distance = pdist([a_features[i].tolist(), b_features[i].tolist()], metric='correlation')
            a_to_c_distance = pdist([a_features[i].tolist(), c_features[i].tolist()], metric='correlation')

            if a_to_b_distance < a_to_c_distance:
                predictions.append('1')
            else:
                predictions.append('0')

    predictions = "\n".join(predictions)

    with open("/Users/andreidm/ETH/courses/iml-tasks/project_4/res/distance_based_predictions_X.txt", 'w') as file:
        file.write(predictions)

    print("predictions saved")


if __name__ == "__main__":

    # generate_train_and_test_datasets()
    # DISTANCE BASED ACCURACY
    # compute_distances_on_train_set()
    # predict_similarity_based_on_distance()

    train_chunk = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_X_1.csv")
    train_features = train_chunk.sample(frac=0.3, replace=False)

    for i in range(2,13):
        train_chunk = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_data_X_{}.csv".format(i))
        train_features = pandas.concat([train_features, train_chunk.sample(frac=0.3, replace=False)])

    print("training set composed")

    features = train_features.iloc[:, 2:]
    classes = train_features['class']

    # split
    X_train, X_val, y_train, y_val = train_test_split(features, classes, stratify=classes, random_state=constants.SEED)

    start = time.time()

    # get trained model
    best_model = train_xgb(X_train, y_train, X_val, y_val)

    # predict on all test chunks
    predictions = []
    for i in range(1,7):
        test_chunk = pandas.read_csv("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_data_X_{}.csv".format(i))
        chunk_predictions = best_model.predict(test_chunk.iloc[:, 1:], ntree_limit=best_model.best_ntree_limit)

        predictions.extend(chunk_predictions.tolist())

    predictions = "\n".join([str(prob) for prob in predictions])
    with open("/Users/andreidm/ETH/courses/iml-tasks/project_4/res/xgboost_predictions_X_3.txt", 'w') as file:
        file.write(predictions)

    print("xgb predictions saved")

