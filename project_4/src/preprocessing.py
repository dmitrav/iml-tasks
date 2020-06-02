
from PIL import Image
from tqdm import tqdm
import os

PATH_TO_RAW_DATA = '/Users/andreidm/ETH/courses/iml-tasks/project_4/data/food/'
PATH_TO_TRAIN = '/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train/'
PATH_TO_TEST = '/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test/'

TRAIN_TRIPLETS = "/Users/andreidm/ETH/courses/iml-tasks/project_4/data/train_triplets.txt"
TEST_TRIPLETS = "/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_triplets.txt"


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


def generate_train_and_test(size):

    with open(TRAIN_TRIPLETS) as file:
        train_triplets = file.readlines()

    for triplet in tqdm(train_triplets):

        name_a, name_b, name_c = [image_id.replace("\n", "") + ".jpg" for image_id in triplet.split(" ")]

        # read images from triplet and resize them
        image_a = Image.open(PATH_TO_RAW_DATA + name_a).resize((size, size))
        image_b = Image.open(PATH_TO_RAW_DATA + name_b).resize((size, size))
        image_c = Image.open(PATH_TO_RAW_DATA + name_c).resize((size, size))

        # concatenate images horizontally
        class_1_image = get_concat_h(image_a, image_b, image_c)
        class_0_image = get_concat_h(image_a, image_c, image_b)

        # save to different folders
        class_1_image.save(PATH_TO_TRAIN + 'class_1/' + "_".join([name_a, name_b, name_c]) + ".jpg")
        class_0_image.save(PATH_TO_TRAIN + 'class_0/' + "_".join([name_a, name_c, name_b]) + ".jpg")

    with open(TEST_TRIPLETS) as file:
        test_triplets = file.readlines()

    for triplet in tqdm(test_triplets):

        name_a, name_b, name_c = [image_id.replace("\n", "") + ".jpg" for image_id in triplet.split(" ")]

        # read images from triplet and resize them
        image_a = Image.open(PATH_TO_TEST + name_a).resize((size, size))
        image_b = Image.open(PATH_TO_TEST + name_b).resize((size, size))
        image_c = Image.open(PATH_TO_TEST + name_c).resize((size, size))

        # concatenate images horizontally
        new_image = get_concat_h(image_a, image_b, image_c)

        # save to different folders
        new_image.save(PATH_TO_TRAIN + 'class_1/' + "_".join([name_a, name_b, name_c]) + ".jpg")


if __name__ == '__main__':

    generate_train_and_test(160)