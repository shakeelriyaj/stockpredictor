import numpy as np #used for numerical computing
import pandas as pd #used for data tables and analysis
import matplotlib.pyplot as plt #used for plotting
from sklearn.preprocessing import MinMaxScaler #used for scaling inputs between 0 and 1
from keras.models import Sequential #neural network layer builder
from keras.layers import Dense, LSTM, Dropout #dense is NNL, long short term memory for sequence modeling, dropout prevents overfitting by randomly dropping neurons.

data = pd.read_csv('GOOGtraindata.csv')
print(data.head()) #shows the first 5 rows of data
print(data.info()) #shows the data types of each column

data["Close"]=pd.to_numeric(data.Close,errors='coerce') #converts the close column to numeric values, if an error, go to NaN
data = data.dropna() #drop all rows that have an NaN values
trainData = data.iloc[:,4:5].values #gets data using integer location indexing, all rows and columns up to index 4 and converts it to numPy array.

print(data.info())

sc = MinMaxScaler(feature_range=(0,1)) #initialize minmaxscaler, data will be scaled from range 0,1
trainData = sc.fit_transform(trainData) #this scales traindata using the minmaxscaler, computes min and max and scales accordingly
print(trainData.shape) #returns the dimensions of the array in the form of (rows, columns)

xTrain = [] #initialize empty list
yTrain = [] #initialize empty list

for i in range(60,1916): #loop iterates through 60 to end
    xTrain.append(trainData[i-60:i,0]) #appends trainData to xTrain, a window of 60 elements from trainData and appends it to xTrain
    yTrain.append(trainData[i,0]) #appends a single data point to yTrain, in the first column
#this creates a sequence of 60 data points as inputs(xTrain) and next data point as outputs(yTrain)

xTrain,yTrain = np.array(xTrain),np.array(yTrain) #this converts the lists xTrain and yTrain into numPy arrays

xTrain = np.reshape(xTrain,(xTrain.shape[0],xTrain.shape[1],1)) #reshapes the xTrain array to 3D array for LSTM model. -> (number of samples, timesteps, number of features)

print(xTrain.shape) #checks shape, will show dimensions of the xTrain

model = Sequential() #initializes the model
#we will now add layers to our keras model
model.add(LSTM(units=100, return_sequences=True, input_shape=(xTrain.shape[1],1))) #100 nueron layer, the return_sequences means that this LSTM layer will return sequences rather than 
#a single output for each input sequence, input_shape=(xTrain.shape[1], 1) specifies the input shape. xTrain.shape[1] is the number of time steps in each sequence, and 1 indicates a single feature. 
model.add(Dropout(0.2)) #adds a dropout layers to prevent overfitting, 0.2 means 20% of the neurons will be dropped randomly during training

#we repeat this process to create layers for a deeper LSTM architecture 
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1)) #adds a dense layer, which is a fully connected output layer with 1 unit, used for regression tasks to predict single continuous values
model.compile(optimizer="adam",loss="mean_squared_error") #compiles the model, adam is an optimizer, loss is the loss function used to compute the error

hist = model.fit(xTrain, yTrain, epochs=20, batch_size=32,verbose=2) #trains the model using the fit() method, xTrain is the input data, yTrain is the target output. epochs=20 specifies 
#the number of times the model is trained, batch_size defines the number of samples per gradient, verbose specifices how detailed the train progress will be displayed.

testData = pd.read_csv('GOOGtestdata.csv')
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[60:,0:].values 
#input array for the model
inputClosing = testData.iloc[:,0:].values 
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape
X_test = []
length = len(testData)
timestep = 60
for i in range(timestep,length):  
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_test.shape

y_pred = model.predict(X_test)
print(y_pred)
predicted_price = sc.inverse_transform(y_pred)


plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()