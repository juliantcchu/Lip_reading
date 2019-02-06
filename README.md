# Lip_reading

"prep_data.py" is used for transforming the videos in "data" into numpy arrays and tag them correspondingly. The arrays and tags is stored to X.pickle and y.pickle respectively. 

"train_model.py" trains the model using "X.pickle" and "y.pickle" and saves the model into "all_lips.h5"

"prep_data_train_model.py" does all the above (for convenience), but X.pickle and y.pickle should already be there when you download it so you can just directly run "train_model.py"

"video.py" is just a module for handling the videos

The original video data cannot be uploaded because of the size. However you probably won't need it to train the model cuz X.pickle and y.pickle is already here. You can just directly run "train_model.py". I'll email you the videos if you need it. 



modules used:

tensorflow
keras (I don't know if it is already included in tensorflow)
numpy
tqdm
cv2
matplotlib
pickle
random
