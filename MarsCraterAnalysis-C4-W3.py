# coding: utf-8

"""
Created on Tue June 30 13:05:34 2016

@author: Chris
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.linear_model import LassoLarsCV
from io import BytesIO
from IPython.display import Image

%matplotlib inline

#bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pd.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pd.read_csv('D:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pd.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pd.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with 'No Morphology'
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ','No Morphology')

#Remove any data with NaN values
data2 = data.dropna()
data2.describe()
data2.head(5)

#To make our lasso regression analysis more interpretable, we'll have craters with no morphology as 0, and craters with
#a morphology as 1

def cratermorph(x):
    if x == 'No Morphology':
        return 0
    else:
        return 1
    
data2['CRATER_MORPHOLOGY_BIN'] = data2['MORPHOLOGY_EJECTA_1'].apply(cratermorph)
data2['CRATER_MORPHOLOGY_BIN'] = data2['CRATER_MORPHOLOGY_BIN'].astype('category')
data2.head(5)

#selecting predictor variables and target variables
predictorvariables = data2[['LATITUDE_CIRCLE_IMAGE','LONGITUDE_CIRCLE_IMAGE','DEPTH_RIMFLOOR_TOPOG','NUMBER_LAYERS',
                            'CRATER_MORPHOLOGY_BIN']]

targetvariable = data2['DIAM_CIRCLE_IMAGE']

#now we'll standardize predictors to have a mean=0 and sd=1
predictors = predictorvariables.copy()

#loop through our predictor variables and pre process them
for a in predictors:
    predictors[a]=preprocessing.scale(predictors[a].astype('float64'))

#split data into training and test data sets
predictors_train, predictors_test, target_train, target_test =     train_test_split(predictors,targetvariable,test_size=.3,random_state=123)

print(len(predictors_train))
print(len(predictors_test))

#Lasso regression model
model = LassoLarsCV(cv=10, precompute=False).fit(predictors_train,target_train)

#print variable names and regression coefficients
dict(zip(predictors.columns,model.coef_))

#plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths of Selected Variables')

#plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
            label='Average across the folds',linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.xlim(1.95,4.0)

#MSE from training and test data
training_error = mean_squared_error(target_train,model.predict(predictors_train))
test_error = mean_squared_error(target_test,model.predict(predictors_test))
print('Training data MSE')
print(training_error)
print('Test data MSE')
print(test_error)

#R-squared from training and test data
rsquared_train=model.score(predictors_train,target_train)
rsquared_test=model.score(predictors_test,target_test)
print('Training data R**2')
print(rsquared_train)
print('Test data R**2')
print(rsquared_test)