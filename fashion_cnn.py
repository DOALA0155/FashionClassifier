from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from functions import *

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_val, y_val = x_train[:10000], y_train[:10000]
x_train, y_train = x_train[10000:], y_train[10000:]

y_train, y_val, y_test = categorical_data(y_train, y_val, y_test)
x_train, x_val, x_test = scaling_data(x_train, x_val, x_test)
x_train, x_val, x_test = add_new_dim(x_train, x_val, x_test)

model = define_model(x_train[0].shape, len(y_train[0]))

history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_val, y_val))
plot_history(history.history)

score = model.evaluate(x_test, y_test)

print("Test loss: {:.3f}".format(score[0]))
print("Test Score: {:.3f}".format(score[1]))

model.save("./Models/FashonCnnModel.h5")
