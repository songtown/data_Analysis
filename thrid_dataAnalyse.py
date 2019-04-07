import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

plt.style.use("fivethirtyeight")
sns.set_style({'font.sans_serif':['simhei','Arial']})

#提取所需数据保存到CSV中
def get_data():
	with open('house_price_data.csv','rt') as f1:
		f = open('./house_price.csv', 'wt', encoding='utf-8',newline='')
		df = pd.read_csv('./house_price_data.csv', usecols=['price', 'bedrooms', 'sqft_living', 'grade'])
		df.to_csv('./house_price.csv')
		f.close()
	f1.close()

#分析数据并实现数据可视化
def data_analyse():
	df=pd.read_csv('./house_price.csv')
	#df.info()	#检查数据缺失值情况
	#房价与室内面积关系的可视化
	f,[ax1,ax2]=plt.subplots(1,2,figsize=(20,5))
	sns.distplot(df['sqft_living'],bins=20,ax=ax1,color='r')
	sns.kdeplot(df['sqft_living'],shade=True,ax=ax1)
	sns.regplot(x='sqft_living',y='price',data=df,ax=ax2)
	plt.show()
	#房价与卧室个数关系的可视化
	f,[ax1,ax2]=plt.subplots(1,2,figsize=(20,8))
	sns.barplot(x='bedrooms',y='price',data=df,ax=ax1)
	#房价与grade关系的可视化图
	sns.barplot(x='grade',y='price',data=df,ax=ax1)
	plt.show()
	#绘制多变量之间的对比关系图
	sns.pairplot(df,vars=('price','sqft_living','grade','bedrooms'))
	plt.show()

#数据划分
data=pd.read_csv('./house_price.csv')
columns=['bedrooms','sqft_living','grade','price']
data=pd.DataFrame(data,columns=columns)			#重新摆放位置
prices=data['price']
features=data.drop('price',axis=1)
features=np.array(features)
prices=np.array(prices)
features_train,features_test,prices_train,prices_test=train_test_split(features,prices,test_size=0.2,random_state=0)

#建立模型
def fit_model(X,y):
	cross_validator=KFold(n_splits=10,shuffle=True)
	regressor=DecisionTreeRegressor()
	params={'max_depth':range(1,11)}
	scoring_fnc=make_scorer(performance_metric)
	grid=GridSearchCV(estimator=regressor,param_grid=params,scoring=scoring_fnc,cv=cross_validator)
	grid=grid.fit(X,y)
	return grid.best_estimator_

#计算R2分数
def performance_metric(y_true,y_predict):
	score=r2_score(y_true,y_predict)
	return score

if __name__ == '__main__':
	get_data()
	data_analyse()

#获得最优模型
optimal_reg=fit_model(features_train,prices_train)
print("最理想模型的参数‘max_depth’是{}".format(optimal_reg.get_params()['max_depth']))
predicted_value=optimal_reg.predict(features_test)
r2=performance_metric(prices_test,predicted_value)
print("最优模型在测试数据上R^2分数{:,.2f}".format(r2))

#房价预测
client_data=[[3,1200,8],[4,720,8],[5,2310,7]]
predicted_price=optimal_reg.predict(client_data)
for i,price in enumerate(predicted_price):
	print("第{}位客户，根据您输入的信息预测到的房价为：￥{:,.2f}".format(i+1,price))