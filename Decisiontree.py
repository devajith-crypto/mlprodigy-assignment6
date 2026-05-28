import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data=pd.read_csv("Decision_Tree.csv")
data=data.drop(["Machine"],axis=1)

Label_encoder=LabelEncoder()
data["Temperature_encoded"]=Label_encoder.fit_transform(data["Temperature"])
data["Vibration_encoded"]=Label_encoder.fit_transform(data["Vibration"])
data["Failure_encoded"]=Label_encoder.fit_transform(data["Failure"])

x=data[["Temperature_encoded","Vibration_encoded"]]
y=data["Failure_encoded"]

model=DecisionTreeClassifier(criterion="entropy")
model.fit(x,y)
print("Decision Tree Model trained successfully")

tree.plot_tree(model,feature_names=["Temperature","Vibration"],class_names=["No","Yes"],filled=True)
plt.title("Decision Tree Visualization")
plt.show()

print("Enter the following features regarding machine conditions as 0,1,2:")
print("high=0,low=1,medium=2")

temperature=int(input("Temperature(High/Low/Medium):"))
vibration=int(input("Vibration(High/Low/Medium):"))
prediction=model.predict([[temperature,vibration]])
print("Predicted output:",prediction[0])
if prediction[0]==1:
    print("Machine Failure")
else:
    print("No Machine Failure")