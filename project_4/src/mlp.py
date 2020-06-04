import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from project_4.src import constants


if __name__ == "__main__":

    batch_size = 64
    image_size = (150, 150)

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

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*image_size, 3)),

        tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # model.build(input_shape=(None, *image_size, 3))

    # keras.utils.plot_model(model, show_shapes=True)

    epochs = 20

    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]

    model.compile(optimizer=keras.optimizers.RMSprop(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    model.fit(train_batches, epochs=epochs, callbacks=callbacks, validation_data=val_batches)

    # img = keras.preprocessing.image.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
    # img_array = keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    #
    # predictions = model.predict(img_array)
    # score = predictions[0]
    # print("This image is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score))