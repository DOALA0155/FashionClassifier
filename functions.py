import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout
import numpy as np

def plot_history(score_history):
    acc_history = score_history["accuracy"]
    val_acc_history = score_history["val_accuracy"]
    loss_history = score_history["loss"]
    val_loss_history = score_history["val_loss"]
    x = range(len(acc_history))

    plt.plot(x, acc_history, label="train_accuracy")
    plt.plot(x, val_acc_history, label="test_accuracy")
    plt.legend(loc="best")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(x, loss_history, label="train_loss")
    plt.plot(x, val_loss_history, label="test_loss")
    plt.legend(loc="best")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def define_model(data_shape, class_shape):
    model = Sequential()
    model.add(Conv2D(100, kernel_size=(3, 3), activation="relu", input_shape=data_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(class_shape, activation="softmax"))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def add_new_dim(x_train, x_val, x_test):
    x_train_added = np.expand_dims(x_train, axis=3)
    x_val_added = np.expand_dims(x_val, axis=3)
    x_test_added = np.expand_dims(x_test, axis=3)
    return x_train_added, x_val_added, x_test_added

def scaling_data(x_train, x_val, x_test):
    x_train_scaled = x_train / 255.
    x_val_scaled = x_val / 255.
    x_test_scaled = x_test / 255.
    return x_train_scaled, x_val_scaled, x_test_scaled

def categorical_data(train_labels, val_labels, test_labels):
    y_train = to_categorical(train_labels)
    y_val = to_categorical(val_labels)
    y_test = to_categorical(test_labels)
    return y_train, y_val, y_test
