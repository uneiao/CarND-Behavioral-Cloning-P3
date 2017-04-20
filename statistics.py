import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import pickle

history = pickle.load(open("history.pickle", "rb"))

### print the keys contained in the history object
print(history.keys())

### plot the training and validation loss for each epoch
print(len(history))
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.xticks(range(len(history["loss"])))
plt.grid()
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()

from keras.models import load_model
model = load_model("model.h5")

model.summary()
model.get_config()
