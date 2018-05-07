import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Sequential
from cnn_model import cnn_model_fn


PRETRAINED_MODEL=False
batch_size=30
epochs=50

if PRETRAINED_MODEL:
    # загружаем модель, обучена на CIFAR10 до точности 0.8
    base_model=load_model('saved_keras_model') 
    # отключаем верхние слои отвечающие за классификацию
    base_model.layers.pop()
    base_model.layers.pop()
    #base_model.layers.pop()
    # запрещаем менять коэфиценты у сверточных слоев
    for layer in base_model.layers:
        layer.trainable = False
    # создаем новую модель 
    model = Sequential()
    # на основе сверточной части уже обученной
    model.add(base_model)
    # добавляем обучаемые fully connected слои для классификации
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
else:
    model=cnn_model_fn()


opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'C:/ML/samsung/data/train',  
        target_size=(32, 32), 
        batch_size=batch_size,
        class_mode='binary')  

validation_generator = test_datagen.flow_from_directory(
        'C:/ML/samsung/data/val',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')
    
model.fit_generator(
        train_generator,
        steps_per_epoch=800 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=200 // batch_size)

