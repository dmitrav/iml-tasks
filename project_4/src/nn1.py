
import tensorflow as tf
from tensorflow import keras
from keras import layers
from project_4.src import constants
from matplotlib import pyplot as plt
import numpy


def make_model(input_shape, num_classes):

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # for size in [128, 256, 512, 728]:
    for size in [64]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # x = tf.keras.layers.Activation("relu")(x)
        # x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        # x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=activation)(x)

    return keras.Model(inputs, outputs)


if __name__ == "__main__":

    batch_size = 64
    image_size = (224, 224)

    train_batches = tf.keras.preprocessing.image_dataset_from_directory(
        constants.PATH_TO_TRAIN, labels="inferred", label_mode="binary", class_names=None, color_mode="rgb",
        batch_size=batch_size, image_size=image_size, shuffle=True, seed=constants.SEED, validation_split=0.2,
        subset="training", interpolation="bilinear", follow_links=False
    )

    val_batches = tf.keras.preprocessing.image_dataset_from_directory(
        constants.PATH_TO_TRAIN, labels="inferred", label_mode="binary", class_names=None, color_mode="rgb",
        batch_size=batch_size, image_size=image_size, shuffle=True, seed=constants.SEED, validation_split=0.2,
        subset="validation", interpolation="bilinear", follow_links=False
    )

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    model.summary()

    keras.utils.plot_model(model, show_shapes=True)

    epochs = 50

    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_batches, epochs=epochs, callbacks=callbacks, validation_data=val_batches)

    # img = keras.preprocessing.image.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
    # img_array = keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    #
    # predictions = model.predict(img_array)
    # score = predictions[0]
    # print("This image is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score))