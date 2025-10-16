import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data=pd.read_csv("heart_disease_data.csv")
x=data.drop(columns="target",axis=1)
y=data['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
model=LogisticRegression(max_iter=2000)
model.fit(x_train,y_train)
pickle.dump(model,open('model.pkl','wb'))


