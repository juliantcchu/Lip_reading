import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import video

DATADIR = "data"

CATEGORIES = ["ba", "bu", "en", "fu", "ge", "hao", "jia", "ke", "kou", "lu", "ma", "mi", "ni", "shi", "shi2", "shu", "ta", "tong", "wo", "xi", "xie", "yau", "yun"]

training_data = []


def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                vid1 = video.vid(os.path.join(path,img)) # convert to array
                new_array = vid1.pic3D  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))



import random

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)


print(X[0].reshape(-1, 100, 100, 39))

X = np.array(X).reshape(-1, 100, 100, 39)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


## train_model
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D

##pickle_in = open("X.pickle","rb")  ##already loaded
##X = pickle.load(pickle_in)
X = X.reshape(5520, 100, 100, 39, 1)

##pickle_in = open("y.pickle","rb")
##y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv3D(128, (3, 3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Conv3D(64, (3, 3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(10))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=64 epochs=100, validation_split=0.1)

model.save("all_lips.h5")
