
import PIL, tqdm
import tensorflow as tf
import numpy, pandas
from project_4.src import constants
from tqdm import tqdm


def get_features_from_pretrained_net(model, image):

    img_data = tf.keras.preprocessing.image.img_to_array(image)
    img_data = numpy.expand_dims(img_data, axis=0)

    img_data = tf.keras.applications.inception_v3.preprocess_input(img_data)

    features_3d = model.predict(img_data)

    features_averaged_1d = features_3d[0].mean(axis=0).mean(axis=0)

    return features_averaged_1d


def generate_train_and_test_datasets():

    IMG_SIZE = (350, 240)

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

    train_data.to_csv("/Users/dmitrav/ETH/courses/iml-tasks/project_4/data/train_data.csv", index=False)
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

    test_data.to_csv("/Users/dmitrav/ETH/courses/iml-tasks/project_4/data/test_data.csv", index=False)
    print("test data saved")


if __name__ == "__main__":

    generate_train_and_test_datasets()