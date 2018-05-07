from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

def cnn_model_fn():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu', name='conv_layer1'))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv_layer2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv_layer3'))
   # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv_layer4'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='fc_layer'))
    model.add(Dense(1, activation='sigmoid', name='ouput_layer'))
    return model