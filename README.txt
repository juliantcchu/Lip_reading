"prep_data.py" is used for transforming the videos in "data" into numpy arrays and tag them correspondingly. The arrays and tags is stored to X.pickle and y.pickle respectively. 

"train_model.py" trains the model using "X.pickle" and "y.pickle" and saves the model into "all_lips.h5"

"prep_data_train_model.py" does all the above (for convenience), but X.pickle and y.pickle should already be there when you download it so you can just directly run "train_model.py"

"video.py" is just a module for handling the videos

All data (both training and validation) are saved in "data". "train_model.py" will directly split them into trainig data and validation data when it is training the model. 