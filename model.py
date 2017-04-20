# -*- encoding:utf-8 -*-


import csv
import cv2
import numpy as np
import random


def main():
    lines = []
    with open("driving_log.csv") as dl_csv:
        reader = csv.reader(dl_csv)
        for line in reader:
            lines.append(line)

    #loading data and preprocessing
    images = []
    measurements = []
    correction = 0.2
    random.shuffle(lines)
    for line in lines:
        if random.random() > 1:
            continue
        source_path = line[0]
        filename = source_path.split("/")[-1]
        current_path = "IMG/" + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

        image_left = cv2.imread("IMG/" + line[1].split("/")[-1])
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        measurement_left = float(line[3]) + correction
        images.append(image_left)
        measurements.append(measurement_left)

        image_right = cv2.imread("IMG/" + line[2].split("/")[-1])
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
        measurement_right = float(line[3]) - correction
        images.append(image_right)
        measurements.append(measurement_right)

    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    X_train = np.array(augmented_images)
    X_train = np.resize(X_train, (len(augmented_images), 160, 320, 1))
    y_train = np.array(augmented_measurements)

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D, AveragePooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam

    #model structure
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 1)))
    model.add(Cropping2D(cropping=((65, 25), (0, 0))))

    model.add(Convolution2D(18, 5, 5, subsample=(4, 4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Convolution2D(48, 3, 3, subsample=(2, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(1))

    #training process
    adam = Adam(lr=0.0008)
    model.compile(loss="mse", optimizer=adam)
    history = model.fit(
        X_train, y_train, batch_size=32, validation_split=0.05,
        shuffle=True, nb_epoch=20)

    #save the model
    model.save("model.h5")
    import pickle
    pickle.dump(history.history, open("history.pickle", "wb"))

    model.summary()
    model.get_config()


if __name__ == "__main__":
    main()
