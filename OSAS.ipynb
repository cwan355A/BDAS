# -*- coding: utf-8 -*-


# 01-BU
import pandas as pd
import csv
from datetime import datetime
import sklearn
from matplotlib import pyplot
fileTemperature = 'GlobalTemperatures_orginal.csv'
fileEnergy = 'energy_orginal.csv'

# 02-DU
dft = pd.read_csv(fileTemperature)

dft.info()
dft.describe()
dft['LandAndOceanAverageTemperature'].min()
dft['LandAndOceanAverageTemperature'].max()
dft['LandAverageTemperature'].sum()
dfe = pd.read_csv(fileEnergy)
dfe.info()
dfe.describe()
dfe['CO2_emission'].min()
dfe['CO2_emission'].max()

# 03-DP
print(dfe)
print(dft)
newTemperatureData = [['dt', 'landAverageTemperature', 'LandMaxTemperature',
                       'LandMinTemperature', 'landAndOceanAverageTemperature']]
newTemperatureDataDic = dict()
for index, row in dft.iterrows():

    if pd.isnull(row['LandAndOceanAverageTemperature']) == 0:

        if (row['dt'].find('/') != -1):
            dt_time = datetime.strptime(row['dt'], '%d/%m/%Y')
        else:
            dt_time = datetime.strptime(row['dt'], '%Y-%m-%d')
        time_str = dt_time.strftime('%Y')
        if time_str not in newTemperatureDataDic:
            newTemperatureDataDic[time_str] = [1, row['LandAverageTemperature'],
                                               row['LandMaxTemperature'], row['LandMinTemperature'], row['LandAndOceanAverageTemperature']]
        else:
            newTemperatureDataDic[time_str] = [newTemperatureDataDic[time_str][0]+1,
                                               newTemperatureDataDic[time_str][1] +
                                               row['LandAverageTemperature'],
                                               newTemperatureDataDic[time_str][2] +
                                               row['LandMaxTemperature'],
                                               newTemperatureDataDic[time_str][3] +
                                               row['LandMinTemperature'],
                                               newTemperatureDataDic[time_str][4]+row['LandAndOceanAverageTemperature']]

for key in newTemperatureDataDic.keys():

    newTemperatureData.append([key, newTemperatureDataDic[key][1]/newTemperatureDataDic[key][0],
                               newTemperatureDataDic[key][2] /
                               newTemperatureDataDic[key][0],
                               newTemperatureDataDic[key][3] /
                               newTemperatureDataDic[key][0],
                               newTemperatureDataDic[key][4]/newTemperatureDataDic[key][0]])
print('total Records after filtering null value:'+str(len(newTemperatureData)))
# save into new CSV
with open('newTemperatureData.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for row in newTemperatureData:
        writer.writerow(row)


newGasEmissionData = [['Year', 'Country', 'CO2_emission',
                       'Energy_consumption', 'Energy_production', 'GDP', 'Population']]
for index, row in dfe.iterrows():
    if row['Country'] == 'World' and row['Energy_type'] == 'all_energy_types':
        newGasEmissionData.append([row['Year'], row['Country'], row['CO2_emission'],
                                   row['Energy_consumption'], row['Energy_production'], row['GDP'], row['Population']])
print('total Records after filtering null value:'+str(len(newGasEmissionData)))
with open('newGasEmissionData.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for row in newGasEmissionData:
        writer.writerow(row)
        
#3.4.	 Data Integration
#Read the processed two datasets and merge them
dfeNew=pd.read_csv('newGasEmissionData.csv')
dftNew=pd.read_csv('newTemperatureData.csv')
mergedData=[['Year','CO2_emission','LandAverageTemperature','LandAndOceanAverageTemperature',
             'LandMaxTemperature','LandMinTemperature','Energy_consumption','Energy_production','GDP','Population']]
for i,rowT in dftNew.iterrows():
    aryUnit=[rowT['dt'],'',round(rowT['landAverageTemperature'],3),round(rowT['landAndOceanAverageTemperature'],3),
             round(rowT['LandMaxTemperature'],3),round(rowT['LandMinTemperature'],3),'',
             '','','']
    for j,rowE in dfeNew.iterrows():
        if rowT['dt'] == rowE['Year']:
            aryUnit[1]=round(rowE['CO2_emission'],3)
            aryUnit[6]=round(rowE['Energy_consumption'],3)
            aryUnit[7]=round(rowE['Energy_production'],3)
            aryUnit[8]=round(rowE['GDP'],3)
            aryUnit[9]=round(rowE['Population'],3)
    mergedData.append(aryUnit)
print(mergedData)
#save mergedData
with open('mergedData.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for row in mergedData:
        writer.writerow(row)
with open('mergedData1980.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for row in mergedData:
        if row[1] is not None and row[1] != "":
            writer.writerow(row)
IntegrationDataFull=pd.read_csv('mergedData.csv')
IntegrationDataFull.describe()
IntegrationData=pd.read_csv('mergedData1980.csv')
IntegrationData.describe()

# 04-DT
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
Y=IntegrationData['LandAndOceanAverageTemperature']
X=IntegrationData.drop(['LandAndOceanAverageTemperature'],axis=1)
y=Y.values
X_5=X[['Year','CO2_emission','LandAverageTemperature','LandMaxTemperature','LandMinTemperature']]
model = LinearRegression()
model.fit(X_5, y)

importance=model.coef_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))  
pyplot.bar([x for x in range(len(importance))], importance)  
pyplot.show()  
IntegrationData.describe()
IntegrationDataFull.LandAndOceanAverageTemperature.plot(kind='hist',bins=20,color="steelblue",edgecolor="black")
IntegrationDataFull.CO2_emission.plot(kind='hist',bins=20,color="steelblue",edgecolor="black")

# 05-DMM
# 06-DMA
import numpy as np


#Linear Model
#Linear between Year and Global Temperature
X=IntegrationData.drop(['LandAndOceanAverageTemperature'],axis=1)
X_1=X[['Year']]

y = IntegrationData['LandAndOceanAverageTemperature'].values
regressor = LinearRegression()
regressor.fit(X_1, y)

print(regressor.coef_)        
print(regressor.intercept_)   
print(regressor.score(X_1, y))  

pyplot.scatter(X_1, y)
pyplot.plot(X_1, regressor.predict(X_1), color='red')
pyplot.show()

import statsmodels.api as sm
model_sm = sm.OLS(y, sm.add_constant(X)).fit()
print(model_sm.summary())

#prediction
from sklearn.model_selection import train_test_split 
x = IntegrationData.iloc[:, :1].values  
print(x)
y = IntegrationData.iloc[:, 3].values 
print(y) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=0)
regressor = LinearRegression()  
regressor.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
regressor = LinearRegression()  
regressor.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
y_pred = regressor.predict(x_test.reshape(-1,1))

pyplot.scatter(x_test,y_test,color='black')
pyplot.plot(x_test,regressor.predict(x_test),color='blue',linewidth=3)
pyplot.title("The Linear Model")
pyplot.show()

accuracy = regressor.score(x_test, y_test)
print("accuracy value：", accuracy)

np.set_printoptions(suppress=True)
influence = model_sm.get_influence()
cooks = influence.cooks_distance
print(cooks)
pyplot.scatter(X[['Year']], cooks[0])
pyplot.xlabel('Year')
pyplot.ylabel('Cooks Distance')
pyplot.show()

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence as olsi
x1 = IntegrationData['Year'].values
y1 = IntegrationData['LandAndOceanAverageTemperature'].values
lm = sm.OLS(y1, x1).fit()
studentized_residuals = olsi(lm).resid_studentized
leverage_pts = olsi(lm).hat_matrix_diag
cook_dist = olsi(lm).cooks_distance
fig, ax = pyplot.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(lm, alpha  = 0.05, ax = ax, criterion="cooks")

#Linear between Gas emmision and Global Temperature
X_2=X[['CO2_emission']]

y = IntegrationData['LandAndOceanAverageTemperature'].values
regressor = LinearRegression()
regressor.fit(X_2, y)

print(regressor.coef_)        
print(regressor.intercept_)   
print(regressor.score(X_2, y))  

pyplot.scatter(X_2, y)
pyplot.plot(X_2, regressor.predict(X_2), color='red')
pyplot.show()


# Time series Model
dates=X[['Year']]
TemperatureVals=IntegrationData['LandAndOceanAverageTemperature'].values
# Plot the time series
pyplot.plot(dates, TemperatureVals)
pyplot.xlabel('Date')
pyplot.ylabel('Temperature')
pyplot.title('Time Series Plot')
pyplot.show()





# 07-DM
#
from sklearn.model_selection import train_test_split

y=IntegrationData['LandAndOceanAverageTemperature']
x=IntegrationData.drop('LandAndOceanAverageTemperature',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print("shape of original dataset :", IntegrationData.shape)
print("shape of input - training set", x_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", x_test.shape)
print("shape of output - testing set", y_test.shape)
IntegrationData=pd.read_csv('mergedData1980.csv')
IntegrationData.describe()

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(IntegrationData['LandAndOceanAverageTemperature'].values)
model_fit = model.fit()
print(model_fit.summary())


residuals = IntegrationData['LandAndOceanAverageTemperature']
fig, ax = pyplot.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
pyplot.show()

model_fit.predict(dynamic=False)
pyplot.show()

# 08-INT
X=IntegrationData.drop(['LandAndOceanAverageTemperature'],axis=1)
X_3=X[['LandAverageTemperature']]

y = IntegrationData['LandAndOceanAverageTemperature'].values
regressor = LinearRegression()
regressor.fit(X_3, y)

print(regressor.coef_)        
print(regressor.intercept_)   
print(regressor.score(X_3, y))  

pyplot.scatter(X_3, y)
pyplot.plot(X_3,regressor.predict(X_3), color='red')
pyplot.show()

pyplot.scatter(IntegrationData['LandAndOceanAverageTemperature'], IntegrationData['Year'])
pyplot.title('Scatter Plot')
pyplot.xlabel('LandAndOceanAverageTemperature')
pyplot.ylabel('Year')
pyplot.show()

#8.5.1.	Iteration Repetition
X=IntegrationData.drop(['LandAndOceanAverageTemperature'],axis=1)
X_1=X[['Energy_consumption']]

y = IntegrationData['LandAndOceanAverageTemperature'].values
regressor = LinearRegression()
regressor.fit(X_1, y)

print(regressor.coef_)        
print(regressor.intercept_)   
print(regressor.score(X_1, y))  

pyplot.scatter(X_1, y)
pyplot.plot(X_1, regressor.predict(X_1), color='red')
pyplot.show()

#Error by using IntegrationDataFull.csv
'''
X=IntegrationDataFull.drop(['LandAndOceanAverageTemperature'],axis=1)
X_1=X[['Energy_consumption']]

y = IntegrationDataFull['LandAndOceanAverageTemperature'].values
regressor = LinearRegression()
regressor.fit(X_1, y)

print(regressor.coef_)        
print(regressor.intercept_)   
print(regressor.score(X_1, y))  

pyplot.scatter(X_1, y)
pyplot.plot(X_1, regressor.predict(X_1), color='red')
pyplot.show()
'''




