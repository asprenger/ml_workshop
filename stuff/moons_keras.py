
'''
Use the 'moons' example from the wildml.com blog but implement a simple NN
with a hidden layer in Keras.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2

import matplotlib
import matplotlib.pyplot as plt

def plot_decision_boundary(X, model):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	
    # Generate a grid of points with distance h between them
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Map predictions to 0.0/1.0
    Z = map(lambda x: map(lambda y: 0.0 if y < 0.5 else 1.0, x), Z)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()
	
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)

# X.shape: (200, 2)
# y.shape: (200,)

print "Build and compile a basic model"
model = Sequential()
model.add(Dense(50, input_dim=2, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))
model.summary()

sgd = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# The batch size has an impact on the model performance
model.fit(X, y, batch_size=32, nb_epoch=1000, verbose=1, validation_split=0.1)

scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

plot_decision_boundary(X, model)