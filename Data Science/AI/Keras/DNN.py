import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy



model = Sequential([
    Dense(16,input_shape=(1,), activation="relu"), #input
    Dense(1024,activation="relu"),
    Dense(512,activation="relu"),
    Dense(2,activation="softmax") #output
])

x = []
y = []

optimizer = Adam(lr=".0001")

print(model.summary())

model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x,y,batch_size=10,epochs=20,verbose=2)
