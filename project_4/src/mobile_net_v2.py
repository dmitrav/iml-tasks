
import tensorflow as tf
from project_4.src import constants
from matplotlib import pyplot as plt

if __name__ == "__main__":

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