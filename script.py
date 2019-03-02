from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import matplotlib as mpl
mpl.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

dataset = fetch_mldata('MNIST original')

# Valores entre 0 e 1
data = dataset.data.astype("float") / 255
labels = dataset.target

trainX, testX, trainY, testY = train_test_split(data, labels)


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = Sequential()
model.add(Dense(128, input_shape=(784,), activation="sigmoid"))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

H = model.fit(trainX, trainY, batch_size=128, epochs=500, verbose=2, validation_data=(testX, testY))

predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,10), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,10), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,10), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,10), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


for i in range(10):
    print("##########################################################")
    print("##### Número:    " + str(np.where(predictions[i] == predictions[i].max())[0][0]))
    print("##########################################################")
    plt.imshow(testX[i].reshape((28,28)), cmap="Greys")
    plt.show()
