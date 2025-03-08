from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#verileri alma
cancer = load_breast_cancer()
data=pd.DataFrame(cancer.data,columns=cancer.feature_names)
data["target"]=cancer.target

#feature'lar ve label tanımlama 
X=cancer.data
y=cancer.target 
#sample'ları train ve test olarak ayarlama ,sonra da scale etme
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#model olusturma
model=KNeighborsClassifier(n_neighbors=9)
#model egitimi
model.fit(x_train,y_train)
#model test etme
y_pred=model.predict(x_test)
#doğruluğunu tespit etme
acc=accuracy_score(y_test, y_pred)
print(acc)

cfs=confusion_matrix(y_test, y_pred)
print(cfs)


#Hyperparameter ayarlama
k_values=[]
acc_values=[]

for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    acc=accuracy_score(y_test, y_pred)
    acc_values.append(acc)
    k_values.append(k)
    

plt.figure()
plt.plot(k_values,acc_values,marker="o", linestyle="-")   
plt.title("KNN")
plt.xlabel("k_values")
plt.ylabel("acc_values")
plt.xticks(k_values)
plt.yticks(acc_values)    
plt.grid(True)
    
#k degeri oluşan grafige göre 9 olarak bulunmuştur.Modelde n_neighbors değeri 9 olarak ayarlanmıştır.
    
    