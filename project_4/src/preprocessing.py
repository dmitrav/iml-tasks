
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from project_4.src import constants
import numpy
from matplotlib import pyplot
from keras.preprocessing import image
import os


def get_minimal_image_size():

    min_width, min_height = 1000, 1000  # pictures are smaller

    for file in os.listdir(PATH_TO_RAW_DATA):

        if file.endswith('.jpg'):
            image = Image.open(PATH_TO_RAW_DATA + file)
            width, height = image.size

            if 0 < width < min_width:
                min_width = width
            if 0 < height < min_height:
                min_height = height

    return min_width, min_height  # 354 242


def get_concat_h(im1, im2, im3):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))
    return dst


def generate_train_and_test_images(width, height):

    with open(constants.TRAIN_TRIPLETS) as file:
        train_triplets = file.readlines()

    for triplet in tqdm(train_triplets):

        name_a, name_b, name_c = [image_id.replace("\n", "") for image_id in triplet.split(" ")]

        # read images from triplet and resize them
        image_a = Image.open(constants.PATH_TO_RAW_DATA + name_a + ".jpg").resize((width, height))
        image_b = Image.open(constants.PATH_TO_RAW_DATA + name_b + ".jpg").resize((width, height))
        image_c = Image.open(constants.PATH_TO_RAW_DATA + name_c + ".jpg").resize((width, height))

        # concatenate images horizontally
        class_1_image = get_concat_h(image_a, image_b, image_c)
        class_0_image = get_concat_h(image_a, image_c, image_b)

        # save to different folders
        class_1_image.save(constants.PATH_TO_TRAIN + 'class_1/' + "_".join([name_a, name_b, name_c]) + ".jpg")
        class_0_image.save(constants.PATH_TO_TRAIN + 'class_0/' + "_".join([name_a, name_c, name_b]) + ".jpg")

    with open(constants.TEST_TRIPLETS) as file:
        test_triplets = file.readlines()

    for triplet in tqdm(test_triplets):

        name_a, name_b, name_c = [image_id.replace("\n", "") for image_id in triplet.split(" ")]

        # read images from triplet and resize them
        image_a = Image.open(constants.PATH_TO_RAW_DATA + name_a + ".jpg").resize((width, height))
        image_b = Image.open(constants.PATH_TO_RAW_DATA + name_b + ".jpg").resize((width, height))
        image_c = Image.open(constants.PATH_TO_RAW_DATA + name_c + ".jpg").resize((width, height))

        # concatenate images horizontally
        new_image = get_concat_h(image_a, image_b, image_c)

        # save to different folders
        new_image.save(constants.PATH_TO_TEST + "_".join([name_a, name_b, name_c]) + ".jpg")


def load_and_preprocess_image(path):

  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)

  return image


def get_test_images(img_width, img_height):

    target_size = (img_width, img_height)

    images = []
    for img in os.listdir(constants.PATH_TO_TEST):

        img = os.path.join(constants.PATH_TO_TEST, img)
        img = image.load_img(img, target_size=target_size)
        img = image.img_to_array(img)
        img = numpy.expand_dims(img, axis=0)
        images.append(img)

    # stack up images list to pass for prediction
    images = np.vstack(images)

    return images


if __name__ == '__main__':

    # generate_train_and_test(350, 240)

    all_image_paths = ['/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test/05000_06402_08086.jpg']

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image)

    import matplotlib.pyplot as plt

    for n, image in enumerate(image_ds.take(1)):

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    print()