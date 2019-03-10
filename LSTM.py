'''
a combination of CNN and LSTM

using Conv3D to do 2D convolution in each frame twice, then feeding it into the LSTM, then the output layer

'''


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras import backend as K


batch_size = 4
num_classes = 23
epochs = 15

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
X = X.reshape(-1, 39, 100, 100, 1)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


X = X/255.0


# convert class vectors to binary class matrices
y = keras.utils.to_categorical(y, num_classes)



model = Sequential()

model.add(Conv3D(128, (1, 3, 3), activation = "relu"))
model.add(MaxPooling3D((1, 2, 2)))
model.add(Conv3D(128, (1, 3, 3), activation = "relu"))
model.add(MaxPooling3D((1, 2, 2)))

model.add(Reshape((39, -1)))
model.add(LSTM(128, return_sequences=True, input_shape=(39, -1)))

model.add(Flatten())
model.add(Dense(num_classes, activation = "softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)