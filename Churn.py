import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import f1_score

#Importing the data
X = pd.read_csv('Churn.csv')
print(X)

#Checking for missing values
print(X.isnull().sum())


# Performing labelencoding 
vars_X = [ 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'MonthlyCharges', 'TotalCharges',
'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']  

le = LabelEncoder()
for i in vars_X:
	X[i] = le.fit_transform(X[i])

f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(X.corr(),annot = True, linewidths = 0.4, fmt = '.1F')
plt.show()	

# making churn the target column
y = X['Churn']	

# Dropping variables which does not matter
X = X.drop([ 'Churn','customerID','gender', 'SeniorCitizen'],axis=1)
print(X)

print(X.columns)

print(X)
print(y)

# Incluidng features in model which matter
X = X[[ 'MonthlyCharges', 'TotalCharges', 'tenure', 'Contract', 'TechSupport', 'PaymentMethod']]

# Preprocessing the document and converting into float values
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Hndling infinitelylarge value or any Nan's
np.nan_to_num(X)

# Performing cross cvalidation on the training set 
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Fitting a random forest model to the data
rf = RandomForestClassifier(criterion='entropy')
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

print("Random Forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))

# computing a confusion matrix for calculating f-score
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
np.set_printoptions(precision=2)
print (classification_report(y_test, y_pred))


# Feature importance
feature_imp = pd.Series(rf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print(feature_imp)

# Visulaizing the importance of the features
sns.barplot(x=feature_imp.index, y=feature_imp)
plt.xlabel('Features')
plt.ylabel('Features Importance Score')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()








