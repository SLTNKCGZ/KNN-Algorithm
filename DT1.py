from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

iris=load_iris()

x=iris.data
y = iris.target
#test size ı sample sayısı büyüdükçe küçült
x_train,x_test,y_train,y_test=train_test_split(x, y ,test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

acc=accuracy_score(y_test, y_pred)
cfs=confusion_matrix(y_test, y_pred)
print(acc)
print(cfs)

plt.figure(figsize=(15,10))
plot_tree(model,filled=True,feature_names= iris.feature_names,class_names=iris.target_names)
plt.show()

feature_importance=model.feature_importances_
feature_names=iris.feature_names
sorted_values=sorted(list(zip(feature_importance,feature_names)),reverse=True)
for importance,name in sorted_values:
    print(f"{name} : {importance}")