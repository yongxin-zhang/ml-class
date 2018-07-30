from keras.datasets import mnist
from keras.models import Sequential # evey layer feeds into next layer, no convolution 
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

import wandb # weights and bias database 
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## here re-org the data into two category problem 5 or not 5
is_five_train = y_train == 5
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
## batch_size is configurable too
model.fit(X_train, is_five_train, epochs=10, batch_size=64, validation_data=(X_test, is_five_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels)])


