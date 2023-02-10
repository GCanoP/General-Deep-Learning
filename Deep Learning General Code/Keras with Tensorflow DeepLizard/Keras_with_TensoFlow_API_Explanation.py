"""
=============================================================================
KERAS WITH TENSORFLOW COURSE
A Practical Keras Network Implementation from Deep Lizard Channel.
author : Gerardo Cano Perea. 
date : January 11, 2021
=============================================================================
A basic implementation for the Keras Environment and the models used to build
a neural network. Keras API runs in TensorFlow 2.0.
"""

# Importing Relevant Packages
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')
np.random.seed(1)

# Establishing the Basic Data.
train_labels = []
train_samples = []

# Creating the Simple Dataset. 
for i in range(50):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)
    # the 5% of older individuals who did not experience side effects.
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)
for i in range(1000):
    # The 95% of younger individuals who did not experience side effects.
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)
    # The 95% of older individuals who did experience side effects.
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# Converting Variables in np.array
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
[train_labels, train_samples] = shuffle(train_labels, train_samples)

# Scaling the train sample set. 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# Importing Keras API Utilities
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

model = Sequential()
model.add(Dense(units=16, input_shape=(1,), activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.summary()

# Compiling the model. 
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the model. 
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=100, shuffle=True, verbose=1)

# Predict from Model.
predict = model.predict(scaler.transform((np.array(85)).reshape(-1, 1)))
print(predict)
