
import PIL
from keras.preprocessing import image
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy

model = VGG16(weights='imagenet', include_top=False)
model.summary()

img_path = '/Users/andreidm/ETH/courses/iml-tasks/project_4/data/food/00000.jpg'

img = image.load_img(img_path, target_size=(224, 224))

img_data = image.img_to_array(img)
img_data = numpy.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_feature = model.predict(img_data)

print(vgg16_feature.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(vgg16_feature)

print(feature_batch_average.shape)
