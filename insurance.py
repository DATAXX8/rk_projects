import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# importing the training data
X = pd.read_csv('insurance.csv')
X

# importing the testing set
Y = pd.read_csv('test.csv')
Y

# dimensions of the training data set
print(X.shape)

#checkin for missing values in the daatset
missing = print(X.isnull().sum())

#Looking at the data first 200 lines
print(X.head(200))
print(X.columns)
print(X['Insurance_History_5'])
print(X['Employment_Info_6'])
print(X['Employment_Info_4'])
print(X['Employment_Info_1'])

# Printing all the columns for the data
for i in X.columns:
	print(i)

# Fiiling in the mssing values in the dataset
fill_data = X.fillna(method='bfill',axis=0).fillna(0)
print(fill_data)

X = fill_data
# Calculating value of all non-null objects 
print(X.isnull().sum())

fill_Y = Y.fillna(method='bfill',axis=0).fillna(0)
fill_Y

Y = fill_Y

# Understanding the data
print(X.describe())

# Designating y as the taget column in the training set
y = X['Response']

# dropping unnecessary columns
X = X.drop(['Response','Id'],axis=1)

print(X)

Y = Y.drop(['Id'],axis=1)

# Using label encoder to decode the product column
var_X = ['Product_Info_2']
le = LabelEncoder()
for i in var_X:
    X[i] = le.fit_transform(X[i])


var_Y = ['Product_Info_2']
le = LabelEncoder()
for i in var_Y:
    Y[i] = le.fit_transform(Y[i])

# Extracting the right amount of data

X = X[['BMI','Wt','Product_Info_4','Medical_History_15','Ins_Age','Medical_History_4','Family_Hist_3','Employment_Info_1','Family_Hist_4','Family_Hist_2','Family_Hist_5','Employment_Info_6','Insurance_History_5']]

Y = Y[['BMI','Wt','Product_Info_4','Medical_History_15','Ins_Age','Medical_History_4','Family_Hist_3','Employment_Info_1','Family_Hist_4','Family_Hist_2','Family_Hist_5','Employment_Info_6','Insurance_History_5']]

# preprocessing the data using StandardScaler(Numpy array)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

Y = preprocessing.StandardScaler().fit(Y).transform(Y.astype(float))

# slicing the Test set to reshaape the array 
Y = Y[0:17815]

# Performing cross-validation with 70% training and 30% testing
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Handling Nan's or any infinitely large values
np.nan_to_num(X)
np.nan_to_num(Y)

# Fitting a random forest model to measure feature importance engineering
rf = RandomForestClassifier(criterion='entropy')
rf.fit(X_train,y_train)
y_pred = rf.predict(Y)
print("Random Forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))

#Applying logistic regression to the function
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(Y)
print("Logistic regression ", metrics.accuracy_score(y_test,y_pred))

# Feature importanceenginnering after fitting a random forest model to the training data.
feature_imp = pd.Series(rf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print(feature_imp)

#Visulaizing the importance of the variable
sns.barplot(x=feature_imp.index, y=feature_imp)

plt.xlabel('Features')
plt.ylabel('Features Importance Score')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()



