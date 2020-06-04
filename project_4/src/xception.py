
import tensorflow as tf
from tensorflow import keras
import numpy
from project_4.src import constants
from matplotlib import pyplot as plt

if __name__ == "__main__":

    batch_size = 64
    image_size = (160, 160)

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

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    base_model = tf.keras.applications.Xception(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(*image_size, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top.

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = data_augmentation(inputs)

    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.), the normalization layer
    # does the following, outputs = (inputs - mean) / sqrt(var)
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    mean = numpy.array([127.5] * 3)
    var = mean ** 2

    # Scale inputs to [-1, +1]
    x = norm_layer(x)
    norm_layer.set_weights([mean, var])

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    # model.summary()
    #
    # # compile initial model
    # model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    #               loss="binary_crossentropy",
    #               metrics=["accuracy"])
    #
    # model.fit(train_batches, epochs=3, validation_data=val_batches)

    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 125

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # recompile
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),  # lower learning rate
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # from keras import backend as K
    # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    # session = tf.compat.v1.Session(config=config)
    # K.set_session(session)

    # fit
    model.fit(train_batches, epochs=20, validation_data=val_batches)

    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']

    loss = model.history['loss']
    val_loss = model.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()