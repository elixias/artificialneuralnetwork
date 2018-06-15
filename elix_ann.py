import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

#data preprocessing, converting categorical values to dummy vars
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1]) #geography
labelencoder_G = LabelEncoder()
X[:,2] = labelencoder_G.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1]) #dummy variables
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#test split
from sklearn.model_selection import train_test_split #change cross_validation to model_selection
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#scaling of inputs
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#forming the model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
"""
#classifier.fit(X_train,y_train,batch_size=10,epochs=10,verbose=1)
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) #!!!
#print results as confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
#predicting result for new customer
customer = sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]]))
print(customer)
print((classifier.predict(customer)>0.5))
"""

#import keras classifier for kfold 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    #layer 1 with dropout
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    classifier.add(Dropout(p=0.1))#disable 10% of neurons
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    #loss='categorical_crossentropy' for more than 2 cats. both are logarithmic losses.
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=1)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train,cv=10,n_jobs=-1) #accs returned by kfold
print("Accuracies",accuracies)
print("Mean",accuracies.mean())
print("Variance",accuracies.std())

#performing gridsearch for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    #layer 1 with dropout
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    classifier.add(Dropout(p=0.1))#disable 10% of neurons
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    #loss='categorical_crossentropy' for more than 2 cats. both are logarithmic losses.
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
    'batch_size':[25,32],
    'epochs':[10,50],
    'optimizer':['adam','rmsprop']
}
gridsearch = GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)
gridsearch = gridsearch.fit(X_train,y_train)
best_params = gridsearch.best_params_
best_accuracy = gridsearch.best_score_
print(best_params,best_accuracy)