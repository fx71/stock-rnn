# Recurrent Neural Network

# Part 0 constants
epochs = 2
units = 50
dropout = 0.2
historyReads = 60

shape = 3

#loads previous model before fit
loadModel = False
loadPathIndex = 2
loadPathPrefix = './checkpoints3/my_checkpoint_'

#save model after training
saveModel = False
savePathIndex = loadPathIndex + 1
savePathPrefix = './checkpoints3/my_checkpoint_'


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:shape+1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(historyReads, 1258):#calculate dynamic
    X_train.append(training_set_scaled[i-historyReads:i, 0:shape])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = units, return_sequences = True, input_shape = (X_train.shape[1], shape)))
regressor.add(Dropout(dropout))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = units, return_sequences = True))
regressor.add(Dropout(dropout))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = units, return_sequences = True))
regressor.add(Dropout(dropout))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = units))
regressor.add(Dropout(dropout))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


#or loading existing weights
if(loadModel):
    regressor.load_weights(loadPathPrefix + str(loadPathIndex))

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = epochs, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:shape].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train.iloc[:, 1:shape+1], dataset_test.iloc[:, 1:shape+1]), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - historyReads:].values
inputs = sc.transform(inputs)
X_test = []
for i in range(historyReads, historyReads+20):
    X_test.append(inputs[i-historyReads:i, 0:shape])
X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)



predicted_stock_price_reshaped = np.array([[predicted_stock_price[0][0], 1, 1]])
for i in range(1, len(predicted_stock_price)):
    temp = np.array([[predicted_stock_price[i][0], 1, 1]])
    predicted_stock_price_reshaped = np.concatenate((predicted_stock_price_reshaped, temp), axis = 0)
    

predicted_stock_price = sc.inverse_transform(predicted_stock_price_reshaped)

# Now real_stock_price and predicted_stock_price has mode dimensions. Additional dimensions 
# must to be cut of before display. 
real_stock_price_display = np.array([[real_stock_price[0][0]]])
for i in range(1, len(real_stock_price)):
    temp = np.array([[real_stock_price[i][0]]])
    real_stock_price_display = np.concatenate((real_stock_price_display, temp), axis = 0)
    
predicted_stock_price_display = np.array([[predicted_stock_price[0][0]]])
for i in range(1, len(predicted_stock_price)):
    temp = np.array([[predicted_stock_price[i][0]]])
    predicted_stock_price_display = np.concatenate((predicted_stock_price_display, temp), axis = 0)

# Visualising the results
plt.plot(real_stock_price_display, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price_display, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#regressor.summary()

#save model to file
if(saveModel):
    regressor.save_weights(savePathPrefix + str(savePathIndex))
