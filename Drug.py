import pandas as pd
import warnings
import pickle

dt=pd.read_csv(r"C:\Users\NATHASHA K\Downloads\drug200.csv")
x = dt[["Age","Sex","BP","Cholesterol","Na_to_K"]]
y = dt.Drug
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y = le.fit_transform(y)
bp=le.fit_transform(dt[["BP"]])
Sex=le.fit_transform(dt[["Sex"]])
Cholesterol=le.fit_transform(dt[["Cholesterol"]])
x["BP"]=bp
x["Sex"]=Sex
x["Cholesterol"]=Cholesterol  
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=10)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(train_x,train_y)
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(test_x)
accuracy = accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
pickle.dump(model,open("Drug.pkl","wb"))