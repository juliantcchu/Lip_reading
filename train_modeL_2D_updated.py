from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras import backend as K

batch_size = 4
num_classes = 23
epochs = 15

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
X = X.reshape(5520, 100, 100, 39, 1)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0


# convert class vectors to binary class matrices
y = keras.utils.to_categorical(y, num_classes)

model = Sequential()
model.add(Conv3D(64, kernel_size=(3, 3, 3),
                 activation='relu'))
model.add(Conv3D(128, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save("all_lips.h5")

##score = model.evaluate(x_test, y_test, verbose=0)
##print('Test loss:', score[0])
##print('Test accuracy:', score[1])