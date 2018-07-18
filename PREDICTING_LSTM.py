import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import json
import time
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error

######################################################################################################
############################################### LSTM #################################################

# 导入数据
# 导入数据
def get_data(filename):
    with open(filename, 'r') as fn:
        f = json.load(fn)
        dates = list(f.keys())
        series = pd.DataFrame(list(f.values()))
        scaler = MinMaxScaler(feature_range=(-1,1)) 
        scaled = scaler.fit_transform(series) 
        nums = scaled.tolist()
    return dates,nums

dates,nums = get_data('date_num_5458.json')
dataset = np.asarray(nums)

# 生成模型训练数据集（确定训练集的窗口长度）
def create_dataset(dataset, look_back): 
	#look_back为窗口，指需要多少数据用来预测下一次的数据，这里取一周的数据，即2016条数据,look_back 就是预测下一步所需要的 time steps
	dataX, dataY = [],[] #X为前一周的数据，Y为待预测的5分钟的数据。
	for i in range(len(dataset) - look_back-1):
		dataX.append(dataset[i:i+look_back])
		dataY.append(dataset[i+look_back]) 
	return np.asarray(dataX), np.asarray(dataY)

# fix random seed for reproducibility 
np.random.seed(7)

# normalize the dataset 
scaler = MinMaxScaler(feature_range=(0, 1)) 
dataset = scaler.fit_transform(dataset) 

# split into train and test sets 
train_size = int(len(dataset) * 0.7) #设定 67% 是训练数据，余下的是测试数据
test_size = len(dataset) - train_size 
train,test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# use this function to prepare the train and test datasets for modeling 
look_back = 2016
trainX, trainY = create_dataset(train, look_back) 
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features] 
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1])) 
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM模型构建
# create and fit the LSTM network 
model = Sequential() 
model.add(LSTM(5, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

start_time_1 = time.time()
model.fit(trainX, trainY, epochs=100, batch_size=50, verbose=2)
end_time_1 = time.time()-start_time_1
print(end_time_1)

# make predictions
start_time_2 = time.time()
trainPredict = model.predict(trainX)
end_time_2 = time.time()-start_time_2
print(end_time_2)
testPredict = model.predict(testX)

# 计算误差
# invert predictions （需要先把预测数据转换成同一单位）
trainPredict = scaler.inverse_transform(trainPredict) 
trainY = scaler.inverse_transform(trainY) 
testPredict = scaler.inverse_transform(testPredict) 
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[0])) 
print('Train Score: %.2f RMSE' % (trainScore)) 
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting 
trainPredictPlot = np.empty_like(dataset) 
trainPredictPlot[:, :] = np.nan 
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict 

# shift test predictions for plotting 
testPredictPlot = np.empty_like(dataset) 
testPredictPlot[:, :] = np.nan 
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict 

# plot baseline and predictions 
plt.plot(nums) 
plt.plot(trainPredictPlot) 
plt.plot(testPredictPlot) 
plt.show()

with open('predicted_svm.json','r') as fn:
	predicted_svm = json.load(fn)
plt.plot(nums[2016:2516], label = 'actual data')
plt.plot(predicted_svm[:500], label = 'svm')
plt.plot(trainPredictPlot[2016:2516], label = 'lstm') 
plt.legend()
plt.show()