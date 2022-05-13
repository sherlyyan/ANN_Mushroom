# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:53:03 2019

@author: maryy
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import time

"""Code Flow
The purpose of the code is to build and train an ANN Classifier using data 
from the "mushrooms.csv" file. The dataset is preprocessed and split into training
and test set. The training set is used to train the ANN model and test set is
used to evaluate the result of model by its accuracy rate.
"""
"""Variables Description
cap-shape, cap-surface, cap-color are cap features.
bruises	odor
gill-attachment, gill-spacing, gill-size, gill-color are grill features.
stalk-shape, stalk-root, stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,	
stalk-color-below-ring are stalk features.
veil-type, veil-color are veil features.
ring-number, ring-type are ring features.
spore-print-color	
population	
habitat
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#read in data
data = pd.read_csv("mushrooms.csv")
#call LabelEncoder
labelencoder=LabelEncoder()
#encode string data to integers
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
#inspect the shape of data
data.shape   #(8124,23)
#split X and y
X = data.iloc[:,1:23]
y = data.iloc[:, 0]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#call StandarScaler
scaler = StandardScaler()
#scale X variables
X = scaler.fit_transform(X)
#set seed to 7
seed = 7
np.random.seed(seed)
#split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#set start time
start_time = time.time()
#build model
model = Sequential()
#input layer with 
model.add(Dense(10,input_dim = 22, activation = 'relu'))
#model.add(Dense(5, activation = 'relu'))
#output layer with 2 classes
model.add(Dense(2, activation = 'softmax'))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#compile the model with rmsprop optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#fit the model using training set
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32, verbose=2)
#set end time
end_time = time.time()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#evaluate the model using train set
scores = model.evaluate(X_train,to_categorical(y_train))
#print training accuracy
print("\ntraining","%s: %.2f" % (model.metrics_names[1], scores[1]))
#evaluate the model using test set
scores = model.evaluate(X_test,to_categorical(y_test))
#print test accuracy
print("\ntest","%s: %.2f" % (model.metrics_names[1], scores[1]))
#print running time
print("--- %s seconds ---" % (end_time - start_time))