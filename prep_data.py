import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import video

DATADIR = "data"

CATEGORIES = ["ba", "bu", "en", "fu", "ge", "hao", "jia", "ke", "kou", "lu", "ma", "mi", "ni", "shi", "shi1", "shu", "ta", "tong", "wo", "xi", "xie", "yau", "yun"]

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        vid1 = video.vid(os.path.join(path,img)) # convert to array
        new_array = vid1.pic3D  # convert to array

        break  # we just want one for now so break
    break  #...and one more!
    
print(new_array)
print(new_array.shape)


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

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)



