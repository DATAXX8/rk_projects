import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv('loan_train.csv')

Y = pd.read_csv('loan_test.csv')


print(X.shape)

print(X.info())
print(X.head(200))

#Dealing with Nan's
missing = X.isnull().sum()
print(missing)

#Dropping missing values (Nan's in pandas)
fill_data = X.fillna(method='bfill',axis=0).fillna(0)
print(fill_data)

fill_y = Y.fillna(method='bfill',axis=0).fillna(0)
print(fill_y)

print(fill_data.info())

#Dropping loan ID before training the model
X = fill_data
drop = X.drop(['Loan_ID'],axis=1)

X = drop
print(X)

Y =fill_y
drooping = Y.drop(['Loan_ID'],axis=1)

Y = drooping
print(Y)

# Plotting the heatmap to see correlation between diferent variables
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(X.corr(),annot = True, linewidths = 0.4, fmt = '.1F')
plt.show()

# Joint plot for understanding bivariate relationship
g = sns.jointplot('ApplicantIncome','LoanAmount',data=X,kind='kde',color='m')
g.plot_joint(plt.scatter,c='g',s=300,linewidth=1,marker='+')
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$","$Y$")
plt.show()

# ploting the regression plot to visualize the data
a = sns.regplot(x ='LoanAmount',y ='ApplicantIncome',data=X)
plt.show()

#Applying label encoding to the dataset cause we have to convrt text to data
X['Gender'].fillna(X['Gender'].mode()[0], inplace=True)
X['Married'].fillna(X['Married'].mode()[0], inplace=True)
X['Dependents'].fillna(X['Dependents'].mode()[0], inplace=True)
X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mode()[0], inplace=True)
X['Credit_History'].fillna(X['Credit_History'].mode()[0], inplace=True)
X['Property_Area'].fillna(X['Property_Area'].mode()[0], inplace=True)
X['Education'].fillna(X['Education'].mode()[0], inplace=True)
X['Self_Employed'].fillna(X['Self_Employed'].mode()[0], inplace=True)

Y['Gender'].fillna(Y['Gender'].mode()[0], inplace=True)
Y['Married'].fillna(Y['Married'].mode()[0], inplace=True)
Y['Dependents'].fillna(Y['Dependents'].mode()[0], inplace=True)
Y['Loan_Amount_Term'].fillna(Y['Loan_Amount_Term'].mode()[0], inplace=True)
Y['Credit_History'].fillna(Y['Credit_History'].mode()[0], inplace=True)
Y['Property_Area'].fillna(Y['Property_Area'].mode()[0], inplace=True)
Y['Education'].fillna(Y['Education'].mode()[0], inplace=True)
Y['Self_Employed'].fillna(Y['Self_Employed'].mode()[0], inplace=True)

#Import preprocessing and label encoder
var_X = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_X:
    X[i] = le.fit_transform(X[i])

var_Y = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_Y:
    Y[i] = le.fit_transform(Y[i])


y = X['Loan_Status']
X = X.drop(['Loan_Status'],axis=1)
Y = Y[0:123]

## very impppp(infinty values,NAN)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
Y = preprocessing.StandardScaler().fit(Y).transform(Y.astype(float))
np.nan_to_num(X)
np.nan_to_num(Y)

X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Fitting a logistic regression model
LR = LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train)
LR
y_pred = LR.predict(Y)
print(LR.score(X_train,y_train))
print(metrics.accuracy_score(y_pred,y_test))

# EVluating using F score
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
np.set_printoptions(precision=2)
print (classification_report(y_test, y_pred))

# Decision tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 1)
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(Y)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

# implementing SVM
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(Y) 

cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
np.set_printoptions(precision=2)
print (classification_report(y_test, y_pred))
print('SVM accuracy:',f1_score(y_test, y_pred, average='weighted')) 

# Performing random forest algorithim(Ensemble mthods are known to increase accuracy)
rf = RandomForestClassifier(criterion = 'entropy')
rf.fit(X_train,y_train)
y_pred = rf.predict(Y)
print("Random Forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))

# Performing feature selection to improve performance of the model
#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(X_train,y_train)
#feature_imp = pd.Series(rf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
#print(feature_imp)


#sns.barplot(x=feature_imp, y=feature_imp.index)
#Add labels to your graph
#plt.xlabel('Feature Importance Score')
#plt.ylabel('Features')
#plt.title("Visualizing Important Features")
#plt.legend()
#plt.show()








