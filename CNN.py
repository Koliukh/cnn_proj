import numpy as np
import os
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential , Model, load_model
from cnn_model import cnn_model_fn
from keras.layers import Dense

TRAIN_MODEL=True
batch_size=50
epochs=50


if TRAIN_MODEL:
# подготавливаем тренировочный и валидационный датасеты
# для тренировочного датасета используется augmentation
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.4,
            zoom_range=0.4,
            horizontal_flip=True)
    #ImageDataGenerator удобен т.к. автоатически сопоставляет метки классов 
    # для данных с учетом структуры каталогов
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            './data/train',  
            target_size=(32, 32), 
            batch_size=batch_size,
            class_mode='binary')  
    
    validation_generator = test_datagen.flow_from_directory(
            './data/val',
            target_size=(32, 32),
            batch_size=batch_size,
            class_mode='binary')

    print("Build cnn_model")
    model=cnn_model_fn()
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    print("Train cnn_model")
    
    # обучение на 800 изображениях и валидация на 200
    model_info=model.fit_generator(
            train_generator,
            steps_per_epoch=800 // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=200 )
    #model.save_weights("cnn_model_w.h5")
    #model.save('cnn_model')
else:
    print("Load pre-trained cnn_model")
    model=load_model('cnn_model') 
    
print("Evaluate cnn_model")
# для проверки качества загружаем тестовые и валидационные данные целиком, 800 и 200 соответственно
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './data/train',  
        target_size=(32, 32), 
        class_mode='binary')  
scores1 = model.evaluate_generator(train_generator,800) 
print("Train accuracy = ", scores1[1])
    
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        './data/val',
        target_size=(32, 32),
        class_mode='binary')
scores2 = model.evaluate_generator(validation_generator,200) 
print("Validation accuracy = ", scores2[1])

