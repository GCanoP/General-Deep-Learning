
"""
=======================================================================================================
APPLYING AN AUTOENCODER FOR FRAUD DETECTION MODEL IN KERAS.
Structure -> Encoder -> Latent Space -> Decoder.
author : Gerardo Cano Perea
date : March 11, 2021
=======================================================================================================
"""

# Importing Relevant Packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Importing the Dataset.
data = pd.read_csv('creditcard.csv')  # [5 rows x 31 columns]
nr_classes = data['Class'].value_counts(sort = True)

# Creating a Bar Chart to Compare.
nr_classes.plot(kind = 'bar', rot = 0)
plt.xticks(range(2), labels =['Normal', 'Fraudulent'])
plt.title('Data Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Scatter Plot for Time Variable.
normal = data[data.Class == 0]
fraudulent = data[data.Class == 1]
plt.scatter(normal.Time/3600, normal.Amount, alpha = 0.5, c = '#19323C', label = 'Normal', s = 3)
plt.scatter(fraudulent.Time/3600, fraudulent.Amount, alpha = 0.5, c = '#F2545B', label = 'Fraudulent', s = 3)
plt.xlabel('Time after first transaction [h]')
plt.ylabel('Amount [Euros]')
plt.legend()
plt.show()

# Histogram Plot for Amount Variable.
bins = np.linspace(200, 2500, 100)
plt.hist(normal.Amount, bins = bins, alpha = 1, density = True, label = 'Normal', color = '#19323C')
plt.hist(fraudulent.Amount, bins = bins, alpha = 0.6, density = True, label = 'Fraudulent', color = '#F2545B')
plt.xlabel('Amount of Transaction [Euros]')
plt.ylabel('Percentages of Transactions [%]')
plt.legend(loc='upper right')
plt.show()

# Implementing an Autoencoder for Anomaly Detection.

# Step 1 Data Preprocessing.
from sklearn.preprocessing import StandardScaler

# Time Variable is not Relevant and Amount Need a Standardization.
data.drop(['Time'], axis = 1, inplace = True)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# Creating the Training Set.
from sklearn.model_selection import train_test_split

[X_train, X_test] = train_test_split(data, test_size = 0.2, random_state = 42)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis = 1)
X_train = X_train.values

Y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis = 1)
X_test = X_test.values

# Building the AutoEncoder Model.
np.random.seed(5)
from keras.models import Sequential
from keras.layers import Dense

dim_input = X_train.shape[1]  # 29
model = Sequential()
model.add(Dense(units = 20, activation = 'tanh', input_shape = (dim_input, )))
model.add(Dense(units = 14, activation = 'relu'))
model.add(Dense(units = 20, activation = 'tanh'))
model.add(Dense(units = 29, activation = 'relu'))
model.summary()

from keras.optimizers import SGD
sgd = SGD(learning_rate = 0.01)
model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['accuracy'])

model.fit(X_train, X_train, epochs = 75, batch_size = 32, shuffle = True, validation_data = (X_test, X_test), verbose = 1)
model.save('Autoencoder_Fraud_Detection.h5')

# Making Predictions and Evaluate the Model.
X_pred = model.predict(X_test)
# Calculating the Mean-Square-Error.
mse = np.mean(np.power(X_test - X_pred, 2), axis = 1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

# Behaviour of Precision/Recall Curves.
# Precision is used to measure Positive False Values.
# Recall is used to measure Negative False Values.
[precision, recall, umbral] = precision_recall_curve(Y_test, mse)
plt.plot(umbral, precision[1:], label = "Precision", linewidth=5)
plt.plot(umbral, recall[1:], label = "Recall", linewidth=5)
plt.title('Precision & Recall for Different Thresholds')
plt.xlabel('Umbral')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

umbral_fixed = 0.75
Y_pred = [1 if e > umbral_fixed else 0 for e in mse]

conf_matrix = confusion_matrix(Y_test, Y_pred)



