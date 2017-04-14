# -*- encoding:utf-8 -*-


import csv
import cv2
import numpy as np


def main():
    lines = []
    with open("driving_log.csv") as dl_csv:
        reader = csv.reader(dl_csv)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split("/")[-1]
        current_path = "IMG/" + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D

    model = Sequential()
    model.add(Cropping2D(cropping=((65, 20), (0, 0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    #model.add(Flatten())
    #model.add(Dense(1))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    model.fit(
        X_train, y_train, validation_split=0.2,
        shuffle=True, nb_epoch=10)

    model.save("model.h5")


if __name__ == "__main__":
    main()
