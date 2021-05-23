import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
#Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “
#unpickling” is the inverse operation,whereby a byte stream (from a binary file or bytes-like object) 
#is converted back into an object hierarchy.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('taxi1.csv')
# print(data.head())

data_x = data.iloc[:,0:-1].values #will take all row data exclude last
data_y = data.iloc[:,-1].values
print(data_y)

X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)
#0.3 means 30% data included in testing set and 70% data into training set 
#Random state ensures that the splits that you generate are reproducible. Scikit-learn uses random permutations to generate the splits. 
#Random state ensures that the splits that you generate are reproducible.
reg = LinearRegression()
reg.fit(X_train,y_train) #fit is method and it takes training data

print("Train Score:", reg.score(X_train,y_train)) #97%
print("Test Score:", reg.score(X_test,y_test))    #94%

pickle.dump(reg, open('taxi1.pkl','wb')) #write in binary

model = pickle.load(open('taxi1.pkl','rb')) #read in binary

