import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
import match
from sklearn.metrics import mean_squared_error

a = pd.read_csv(“GOOG.csv”)

df = pd.DataFrame(a)

df.head()

df.tail()

df.shape

df.dtypes

df.describe()

df.info()

df.isnull().sum()

df.duplicated().any(axis=0)

df1 = df.reset_index()[‘Close’]

df1.head()

df1.shape

df1.head()

print(x_train.shape), print(y_train.shape)
plt.plot(df1)
plt.show()

scaler = MinMaxScaler(feature_range = (0,1))
df1 =scaler.fit_transform(np.array(df1).reshape(-1,1))

df1

len(df1)

##splitting dataset into train and test split
trainging_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

training_size,test_size

train_data

plt.plot(train_data)
plt.show()

test_data

plt.plot(test_data)
plt.show()

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
datax, datay = [], []
for i in range(len(dataset)-time_step-1):
	a = dataset[i[i+time_step),0]    ###i=0,  x=0, 1,2,3------99    Y=100
	dataX.append(a)
	dataY.append(dataset[i + time_step, 0])
return np.arrayd(dataX), np.array(dataY)

# reshape into X=t, t+1, t+2..t+99 and Y=t+100

time_step = 100
x_train, y_tain = create_dataset(train_data, time_step)
x_test, ytest = create_dataset(test_data, time_step)

print(x_train.shape), print(y_train.shape)

print(x_test.shape), print(ytest.shape)

# reshape input to be [samples, time step, features] which is required for LSTM
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=sequential()
model.add(LSTM(50,return_seqiences=True,input_shape=(100,1)))
model.add(LSTM(50,return_seqiences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss=’mean_squared_error’ ,optimizer=’adam’)

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to orginal form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

##Transformback to orginal form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE
math.sqrt(mean_squared_error(y_train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(y_test_predict))

### Plotting
# shift train prediction for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

len(test_data)

x_input=test_data[340:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

temp_input

# demonstrate prediction for next 30 days
from numpy import array
1st_output=[]
n_steps=100
i=0
while(i<30):
	if(len(temp_input)>100:
	#print(temp_input)
	x_input=np.aray(temp_input[1:])
	print(“{} day input {}”.format(I,x_input))
	x_input=x_input.reshape(1,-1)
	x_input = x_input.reshape((1, n_steps, 1))
	#print(x_input)
	yhat = model.predict(x_input, verbose=0)
	print(“{} day output {}”.format(i,yhat))
	temp_input.extend(yhat[0].tolist())
	temp_input=temp_input[1:]
	#print(temp_input)
	1st_output.extend(yhat.tolist())
	I=i=1
	else:
	x_input = x_input.reshape((1, n_step,1))
	yhat = model.predict(x_input, verbose=0)
	print(yhat[0])
	temp_input.extend(yhat[0].tolist())
	print(len(temp_input))
	1st_output.extend(yhat.tolist())
	i=i+1
print(1st_output)

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

len(df1)

Scaler.inverse_transform(1st_output)

plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
plt.plot(day_pred,scaler.inverse_transform(1st_output))
plt.savefig(‘30daypredict.png’)

df3=df1.tolist()
df3.extend(1st_output)
plt.plot(df3[1200:])
plt.show()

df3=scaler.inverse_transform(df3).tolist()

plt.plot(df3)
plt.show()

