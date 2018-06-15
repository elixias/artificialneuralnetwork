import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1]) #geography
labelencoder_G = LabelEncoder()
X[:,2] = labelencoder_G.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1]) #dummy variables
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


#change cross_validation to model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#loss='categorical_crossentropy' for more than 2 cats. both are logarithmic losses.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,nb_epoch=10,verbose=1)

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) #!!!

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)