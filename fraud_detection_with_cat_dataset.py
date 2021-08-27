# DATASET URL 
# https://www.kaggle.com/ealaxi/paysim1/code

# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from termcolor import colored as cl # text customization
import itertools # advanced tools

import plotly.graph_objects as go

from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.linear_model import LogisticRegression # Logistic regression algorithm
from sklearn.svm import SVC # SVM algorithm
from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm

from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric

import seaborn as sns

from pprint import pprint

from sklearn import preprocessing


#Metrics Libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Misc libraries
import warnings
warnings.filterwarnings("ignore")

# IMPORTING DATA

df = pd.read_csv('credit_card_fraud.csv', nrows=60000)
# df.drop('Time', axis = 1, inplace = True)

# view first few data
pprint(df.head())


# firstly we take a good look at our dataset with the info function
pprint(df.info())



# # figure out if we have empty data columns and how many
pprint(df.isnull().sum())


# plot heat map to review correlations between features
# plt.figure(figsize=(7,7))
# sns.heatmap(df.corr(),annot=True, fmt='.2f')
# plt.show()


# drop old balance
df.drop(['oldbalanceOrg'], axis = 1, inplace = True)

# #Adding some color to our pivot table 
# color_map = sns.light_palette("blue", as_cmap=True)
# df_pivot.style.background_gradient(cmap=color_map)

# checking for imbalance
# fig = go.Figure(data=[go.Pie(labels=['Not Fraud','Fraud'], values=df['isFraud'].value_counts())])
# fig.show()
# plt.show(); 

#First we need to get the maximum size of isFraud column
max_size = df['isFraud'].value_counts().max()

#we use that to balance our dataset with the code below
# lst = [df]
# for class_index, group in df.groupby('isFraud'):
#     lst.append(group.sample(max_size-len(group), replace=True))
# df = pd.concat(lst)

# convert categorical data into numeric values: 

df = pd.concat([df,pd.get_dummies(df['type'], prefix='type_')],axis=1)
df.drop(['type', 'nameOrig', 'nameDest', 'isFlaggedFraud'],axis=1,inplace = True)

# # Next we normalize our numerical features

# normalize our data manually by dividing the min/max and multiplying by the range 
# in this case we multiply by 1 as we want our data to be between 0 and 1
df=((df-df.min())/(df.max()-df.min()))*1
pprint(df.isnull().sum())

#Splitting dependent and independent variable
X=df.drop('isFraud',axis=1)
y=df['isFraud']

# # Splitting our data into training and testing dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

# commented this out because it's a better idea to balance only the training dataset as opposed to both the training and  testing dataset
# handle balancing of training dataset to avoid oversampling 

# # Create a pivot table with fraud and isflagged fraud
# X_train =pd.pivot_table(X_train,index=["type"],
#                                values=['isFlaggedFraud'],
#                                aggfunc=[np.sum], margins=True)
# # set option up to show values without exponential
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
# pprint(X_train)

sm = SMOTE(random_state=27)
X_train, y_train = sm.fit_resample(X_train, y_train)
pprint(X_train)

print(X_train.shape, y_train.shape)

#Checking the balanced target
fig = go.Figure(data=[go.Pie(labels=['Not Fraud','Fraud'], values=y_train.value_counts())])
fig.show()

# uncomment this section to test with other classification algorithms
# # Building model: 
mlp_cv = MLPClassifier()
rf_cv=RandomForestClassifier(random_state=123)
dt_cv=DecisionTreeClassifier(random_state=123)
svc_cv=SVC(kernel='linear',random_state=123)
knn_cv=KNeighborsClassifier()

cv_dict = {0: 'Neural Network', 1: 'Random Forest',2:'Decision Tree',3:'SVC',4:'KNN'}
cv_models=[mlp_cv,rf_cv, dt_cv, svc_cv, knn_cv]


# cv_dict = {0: 'Random Forest Classifier'}
# cv_models = [rf_cv]

for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X_train, y_train, cv=10, scoring ='accuracy').mean()))

rf_cv.fit(X_train, y_train)
#Predict with the selected best parameter
y_pred = rf_cv.predict(X_test)

# Model Evaluation 
print(classification_report(y_test, y_pred, target_names=['Not Fraud','Fraud']))
