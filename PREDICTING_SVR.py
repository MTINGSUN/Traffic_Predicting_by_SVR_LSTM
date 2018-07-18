import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import json
import time
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


######################################################################################################
############################################### PREDICTION ###########################################
######################################################################################################

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

######################################################################################################
############################################### SVR ##################################################
def svr_predictor(dates, nums, x):
    dates = np.reshape(dates,(len(dates),1))
    svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma=0.001)
    # fitting
    start_time_1 = time.time()
    svr_rbf.fit(dates,nums)
    end_time_1 = time.time() - start_time_1
    # predicting
    start_time_2 = time.time()
    predict = svr_rbf.predict(x)[0]
    end_time_2 = time.time() - start_time_2
    
    return predict,end_time_1,end_time_2

nrow = int(len(dates))
nrow = 3016
predicted_svm,Fitting_time_svm,Predicting_time_svm = [],[],[]
for i in range(2016,nrow):
	print('%d/%d' %(i,nrow))
	predicted,fit_time,predict_time = svr_predictor(dates[i-2016:i], nums[i-2016:i], dates[i]) #用前2016条数据预测后一条数据，即用一周的数据预测下一五分钟的出租车数量
	Fitting_time_svm.append(fit_time) #记录每次fit的时间
	Predicting_time_svm.append(predict_time) #记录每次预测的时间
	print(fit_time, predict_time)
	predicted_svm.append(predicted) #记录每次的预测结果

# Fitting_time_svm存储结果
Fitting_time_svm_file = json.dumps(Fitting_time_svm)
f = open('Fitting_time_svm.json', 'w')
f.write(Fitting_time_svm_file)
f.close()

# Predicting_time_svm存储结果
Predicting_time_svm_file = json.dumps(Predicting_time_svm)
f = open('Predicting_time_svm.json', 'w')
f.write(Predicting_time_svm_file)
f.close()

# predicted_svm存储结果
predicted_svm_file = json.dumps(predicted_svm)
f = open('predicted_svm.json', 'w')
f.write(predicted_svm_file)
f.close()

# plot baseline and predictions 
plt.plot(nums) 
plt.plot(predicted_svm)
plt.show()