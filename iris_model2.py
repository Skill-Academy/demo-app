import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv('iris.csv')
x = df.iloc[:,:-1]
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x,y)

log_reg = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_split=15)
rf_model = RandomForestClassifier(n_estimators=70,criterion='gini',
                                max_depth=4,min_samples_split=15)

log_reg = log_reg.fit(x_train,y_train)
dt_model = dt_model.fit(x_train,y_train)
rf_model = rf_model.fit(x_train,y_train)

pickle.dump(log_reg,open('lr_model1.pkl','wb'))
pickle.dump(dt_model,open('dt_model1.pkl','wb'))
pickle.dump(rf_model,open('rf_model1.pkl','wb'))
