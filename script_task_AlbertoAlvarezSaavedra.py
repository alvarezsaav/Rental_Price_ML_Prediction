import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import sys
import sklearn

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
label_encoder=LabelEncoder()

import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib

from sklearn.model_selection import  train_test_split



##### HOW TO RUN IT ######

# Run this file in the same folder as the dataset and the ML XGBoost models. 


#### path_files = '/content/gdrive/MyDrive/Colab Notebooks/Zoi_Task/'

#path_files=[]


## We load both models: 

loaded_model_1 = joblib.load('XGB_models/' + 'xgb_model_case1_GSCV.pkl')
loaded_model_2 = joblib.load('XGB_models/' + 'xgb_model_case2_GSCV.pkl')

## Here we upload the datasets (adapted to each model, preprocessed):
df_model1=pd.read_csv('df_model1.csv')
df_model2=pd.read_csv('df_model2.csv')

df_final = df_model1
df_final_2 = df_model2



####### Training, Test, Validation sets creation:

### MODEL 1#### 

X = df_final.drop(['totalRent'], axis=1)
y = df_final['totalRent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred_final1 = loaded_model_1.predict(X_test)
mae_final1 = mean_absolute_error(y_pred_final1, y_test)
r2_final1= r2_score(y_pred_final1, y_test)


#### MODEL 2 ####

X_2 = df_final_2.drop(['totalRent'], axis=1)
y_2 = df_final_2['totalRent']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)


y_pred_final2 = loaded_model_2.predict(X_test_2)
mae_final2 = mean_absolute_error(y_pred_final2, y_test_2)
r2_final2= r2_score(y_pred_final2, y_test_2)


##### SCREEN RESULTS ######

print(f"Mean Absolute Error (M1): {mae_final1}")
print(f"R² Score (M1): {r2_final1}")

print(f"Mean Absolute Error (M2): {mae_final2}")
print(f"R² Score (M2): {r2_final2}")



#########################3





