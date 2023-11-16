import numpy as np
import pandas as pd

training_data=pd.read_csv("project1.csv")
#print(training_data.describe())
x=training_data.iloc[ : , :-1].values
y=training_data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#build a classification model

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

#Modal training
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
y_prob=classifier.predict_proba(x_test)[:,-1]

#check accurecy 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test ,y_pred))

#prediction test
new_prediction=classifier.predict(sc.transform(np.array([[20,3000]])))

#picking the model and standard scaler in pickle files as binary(wb)
import pickle 
model_file="classifier.pickle"
pickle.dump(classifier,open(model_file,"wb"))

scaler_file="sc.pickle"
pickle.dump(sc,open(scaler_file,"wb"))