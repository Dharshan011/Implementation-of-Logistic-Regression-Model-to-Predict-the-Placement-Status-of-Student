# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Step 1 :
Import the standard libraries such as pandas module to read the corresponding csv file.

## Step 2 :
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

## Step 3 :
Import LabelEncoder and encode the corresponding dataset values.

## Step 4 :
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

## Step 5 :
Predict the values of array using the variable y_pred.

## Step 6 :
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

## Step 7 :
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

## Step 8:
End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DHARSHAN V
RegisterNumber:  212222230031
*/
```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

![Screenshot 2023-09-28 133226](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/61f7df5f-d710-4e55-aa79-1222093a3c07)


![Screenshot 2023-09-28 133233](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/ed06f3f3-ccdc-4bfb-ab9d-d49c491981e5)

![Screenshot 2023-09-28 133238](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/7b6d4b07-c99a-4dc0-accb-49365aeae539)


![Screenshot 2023-09-28 133245](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/e1c3dde4-a7d7-467f-8f1a-e74ee56a350c)


![Screenshot 2023-09-28 133249](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/b93f3480-1906-4bf6-8b83-8cd57bf898c4)


![Screenshot 2023-09-28 133259](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/c280a3c0-051f-4f2a-b655-9d14286ac70d)
![Screenshot 2023-09-28 133306](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/22caf5e4-e8f3-44d0-9fa0-2166490dddf7)

![Screenshot 2023-09-28 133314](https://github.com/Dharshan011/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497491/2ef7ebc8-dd70-4144-be0f-9b3989faac64)



















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
