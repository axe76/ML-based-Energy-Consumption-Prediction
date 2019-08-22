import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

dataset_train = pd.read_csv('PowerAep2.csv')
training_set = dataset_train.iloc[:,1:2].values
print(dataset_train)

sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled[1974])

x_train = []
y_train = []
for i in range(60,2035):
    x_train.append(training_set_scaled[i-60:i, 0])
    #y_train.append(training_set_scaled[i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train[0])
#print(x_train[1])
#print(y_train[0])
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#print(x_train)
print(x_train.shape[1])

model = Sequential()
model.add(LSTM(units = 128, return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 32))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train,y_train,epochs = 350,batch_size = 32)

dataset_test = pd.read_csv('PowerTest.csv')
print(dataset_train)
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['AEP_MW'],dataset_test['AEP_MW']),axis = 0)
#print(dataset_total)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
#print(inputs)
x_test = []
for i in range (60,76):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
#print(x_test)
predicted_stock = model.predict(x_test)
pedicted_stock = sc.inverse_transform(predicted_stock)
#print(predicted_stock)

plt.plot(actual_power, color = 'black', label = 'Actual')
plt.plot(pedicted_power, color = 'green', label = 'Predicted')
