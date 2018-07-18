import pandas as pd 
import os
import json

######################################################################################################
######################################### DATA PROCESSING ############################################
######################################################################################################

# 导入原始数据
file_dir = os.path.abspath('D:/代码存储/20180628_Traffic_Predictiong_by_SVM&LSTM/数据抽取')
f = file_dir + os.sep +'G_201601.csv'
fn = open(f)
data = pd.read_csv(fn)

# 仅取原始数据中的三列
data = data.loc[:,['lpep_pickup_datetime','Pickup_longitude', 'Pickup_latitude']]

# 消0
data = data[~data['Pickup_longitude'].isin([0])]
data = data[~data['Pickup_latitude'].isin([0])]
# 删除密集于-115°的点，仅取经度密集于-75°的点
data = data[data['Pickup_longitude']>-115]

def transform(df,lens,lon_min,lon_range,lat_min,lat_range):
	rows = []
	for i in range(lens):
		row = df.iloc[i].tolist()
		y_m_d = row[0].split(' ')[0]
		h_m_s = row[0].split(' ')[1]
		month = int(y_m_d.split('-')[1])
		day = int(y_m_d.split('-')[2])
		hour = int(h_m_s.split(':')[0])
		minute = int(h_m_s.split(':')[1])
		second = int(h_m_s.split(':')[2])
		date_num = int((((day*24+hour)*60+minute)*60+second)/300-288)+1
		
		#将经纬度区域划分为10000份
		lon_num = str(int(abs(float(row[1])-lon_min)/(lon_range/100))+1) #将经度划分为100个
		lat_num = str(int(float(row[2]-lat_min)/(lat_range/100))+1) #将纬度划分为100个
		lon_lat = lon_num+' '+lat_num
		date_num = str(date_num)
		date_lon_lat = date_num+' '+lon_num+' '+lat_num
		date_num = int(date_num)

		row.append(month)
		row.append(day)
		row.append(hour)
		row.append(minute)
		row.append(second)
		row.append(date_num)
		row.append(lon_lat)
		row.append(date_lon_lat)
		rows.append(row)
	data = pd.DataFrame(rows, columns = ['datetime','longitude','latitude','month','day','hour','minute','second','date_num','lon_lat','date_lon_lat'])
	return data

lon_min = data['Pickup_longitude'].min()
lon_max = data['Pickup_longitude'].max()
lon_range = lon_max-lon_min
lat_min = data['Pickup_latitude'].min()
lat_max = data['Pickup_latitude'].max()
lat_range = lat_max-lat_min
df = transform(data, lens=len(data), lon_min=lon_min, lon_range=lon_range, lat_min=lat_min, lat_range=lat_range)

li = list(set(df['lon_lat'].tolist()))
print(len(li))

# 对列表中的每个元素进行计数
def getlistnum(li):
	li = list(li) 
	li_ = list(set(li)) #list元素去重
	dict_ = {}
	for i in li_:
		dict_.update({i:li.count(i)})
	return dict_

# 选取经纬度编号为(54 58)的区域进行计数
df_5458 = df[df['lon_lat']=='54 58']
date_num_5458 = getlistnum(df_5458['date_num'])
# 保存结果
file = json.dumps(date_num_5458)
f = open('date_num_5458.json','w')
f.write(file)
f.close()

