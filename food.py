from abc import ABC, abstractmethod


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import matplotlib.pyplot as plt

class FoodCoords(ABC):
    @abstractmethod
    def go_network(self):
        pass


import cv2
#from food import FoodCoords
# не использует нейронку. Но! позволяет незашумленное изображение размечать прямоугольником и возвращать координаты углов
class cvlibrary(FoodCoords):
    def go_network(self, photo):
        im=cv2.imread(photo)
        im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        bw=cv2.threshold(im_gray, 19, 255, cv2.THRESH_BINARY)[1]
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        x,y,w,h,s=stats[1,:]
        cv2.rectangle(im, (x, y), ( x+w, y+h),(0, 0, 255), 1)
        cv2.imwrite('name1.png', im)
        cv2.imshow('name1.png', im)
        cv2.waitKey(0)


class FlowersKeras(FoodCoords):
    def go_network(self):
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        data_dir = pathlib.Path(data_dir)
        image_count = len(list(data_dir.glob('*/*.jpg')))
        print(image_count)

        batch_size = 32
        img_height = 180
        img_width = 180

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = train_ds.class_names
        print(class_names)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.Rescaling(1. / 255)

        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixel values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))

        num_classes = len(class_names)

        model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        epochs = 10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                  input_shape=(img_height,
                                               img_width,
                                               3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

        model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        epochs = 15
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )


        sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
        sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

        img = tf.keras.utils.load_img(
            sunflower_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
        )


# Epoch 15/15
# 92/92 [==============================] - 31s 338ms/step - loss: 0.5371 - accuracy: 0.7977 - val_loss: 0.7863 - val_accuracy: 0.7044
# Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg
# 117948/117948 [==============================] - 0s 0us/step
# 1/1 [==============================] - 0s 78ms/step
# This image most likely belongs to sunflowers with a 93.17 percent confidence.
#
# Process finished with exit code 0
# обучил на 15 эпохах, небольшой датасет. цветы. многоклассовая классификация.



# a = cvlibrary()
# a.go_network('tZ7mr.png')

# b = FlowersKeras()
# b.go_network()







