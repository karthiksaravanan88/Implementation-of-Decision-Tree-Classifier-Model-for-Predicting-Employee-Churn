# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score 
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KARTHIK SARAVANAN B
RegisterNumber:  212224230118
*/
```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()
```
```
print("data.info()")
df.info()
```
```
print("data.isnull().sum()")
df.isnull().sum()
```
```
print("data value counts")
df["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()
```
```
print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/ace72e9e-968b-4158-a0cd-8ccf8dc97765)
![image](https://github.com/user-attachments/assets/7f7273bc-8e70-4f41-928b-fc2f8c8711dd)
![image](https://github.com/user-attachments/assets/a37b04a6-e889-4ef9-b94b-5aaefc350d92)
![image](https://github.com/user-attachments/assets/1a4ae631-18ec-49be-9c7b-ee3f52a33cba)
![image](https://github.com/user-attachments/assets/ff9e41b5-b44b-4243-9263-43fc281b4e4c)
![image](https://github.com/user-attachments/assets/ae7efed2-ba24-4537-a2b5-73c533e579fc)
![image](https://github.com/user-attachments/assets/9ab88c85-027b-4b21-91e0-7417de85a070)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
