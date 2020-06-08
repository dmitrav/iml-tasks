

import numpy
import tensorflow as tf
from project_4.src import constants
from matplotlib import pyplot as plt
import time


def train_model():

    version = "v2"

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

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 151

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    base_learning_rate = 1e-4
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.summary()

    epochs = 20
    validation_steps = 20

    loss0, accuracy0 = model.evaluate(val_batches, steps=validation_steps)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    callbacks = [tf.keras.callbacks.ModelCheckpoint("mnv2_" + version + "_at_{epoch}.h5")]

    history = model.fit(train_batches, epochs=epochs, callbacks=callbacks, validation_data=val_batches)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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


if __name__ == "__main__":

    version = "v3"

    batch_size = 64
    image_size = (224, 224)

    train_batches = tf.keras.preprocessing.image_dataset_from_directory(
        constants.PATH_TO_TRAIN, labels="inferred", label_mode="binary", class_names=None, color_mode="rgb",
        batch_size=batch_size, image_size=image_size, shuffle=True, seed=constants.SEED, validation_split=0.25,
        subset="training", interpolation="bilinear", follow_links=False
    )

    val_batches = tf.keras.preprocessing.image_dataset_from_directory(
        constants.PATH_TO_TRAIN, labels="inferred", label_mode="binary", class_names=None, color_mode="rgb",
        batch_size=batch_size, image_size=image_size, shuffle=True, seed=constants.SEED, validation_split=0.25,
        subset="validation", interpolation="bilinear", follow_links=False
    )

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 151

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    base_learning_rate = 1e-5
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.summary()

    # # load latest weights
    latest = '/Users/andreidm/ETH/courses/iml-tasks/project_4/res/weights/mnv2_v3_0.645_at_20.h5'
    model.load_weights(latest)

    # epochs = 20
    # validation_steps = 20
    #
    # loss0, accuracy0 = model.evaluate(val_batches, steps=validation_steps)
    #
    # print("initial loss: {:.2f}".format(loss0))
    # print("initial accuracy: {:.2f}".format(accuracy0))

    with open("/Users/andreidm/ETH/courses/iml-tasks/project_4/data/test_triplets.txt") as file:
        triplets = file.readlines()

    start_time = time.time()

    predictions = []
    for triplet in triplets:

        image_path = constants.PATH_TO_TEST + "_".join(triplet.split(" ")).replace("\n", "") + ".jpg"

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        image = tf.reshape(image, [1, 224, 224, 3])

        image_class = int(model.predict(image)[0][0] > 0.5)

        predictions.append(image_class)

        # print(image_class)

    print("predictions took", (time.time() - start_time) // 60, "minutes")

    all_predictions = "\n".join([str(prob) for prob in predictions])
    with open("/Users/andreidm/ETH/courses/iml-tasks/project_4/res/mnv2_v3_0.645_at_20.txt", 'w') as file:
        file.write(all_predictions)





    # callbacks = [tf.keras.callbacks.ModelCheckpoint("mnv2_" + version + "_at_{epoch}.h5")]
    #
    # # train further
    # history = model.fit(train_batches, epochs=epochs, callbacks=callbacks, validation_data=val_batches)
    #
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()), 1])
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.ylabel('Cross Entropy')
    # plt.ylim([0, 1.0])
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()
