import numpy as np
import os
import shutil

import matplotlib.pyplot
from matplotlib.pyplot import imshow
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import NearestNeighbors

from keras.utils.np_utils import to_categorical
from cnn_model import cnn_model_fn
from keras.optimizers import SGD

from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.decomposition import PCA


batch_size=50
epochs=50
TRAIN_MODEL=True
pca_comp=300
n_neighbors=15
metric='euclidean'#'euclidean' #'cosine'

input_dir = ".\heap"
output_dir = ".\out"
os.makedirs(output_dir, exist_ok=True)

print("Clear output dir")
for root, dirs, files in os.walk(output_dir):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))


if TRAIN_MODEL:
    # для тренировочного датасета используется augmentation
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True)
    train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

# подготавливаем тренировочный и валидационный датасеты
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
    

# строим модель
    print("Build cnn_model")

    model=cnn_model_fn()

    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# компилируем
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
# обучаем
    model.fit_generator(
            train_generator,
            steps_per_epoch=800 // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=200 // batch_size)
        
    print("Train cnn_model")

    model.save_weights("cnn_model1_w.h5")
    model.save('cnn_model1')
else: # грузим сохраненную
    print("Load pre-trined cnn_model")
    model=load_model('cnn_model1') 

## берем выходы модели перед последним слоем , они будут содержать фичи
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc_layer").output)

# 
listing = os.listdir(input_dir) 

print("Apply cnn_model to extract features")
# собираем выходы с предполеднего слоя в массив фич
features = []
for file in listing:
    img = image.load_img(input_dir+ '\\' + file, target_size=(32, 32))
    x = image.img_to_array(img)
    x=x[ np.newaxis,:,:,:]
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
# уменьшим размерность фич
print("Apply PCA from dim="+str(len(feat))+' to dim='+str(pca_comp))
features = np.array(features)
pca = PCA(n_components=pca_comp)
pca.fit(features)
pca_features = pca.transform(features)
#создаем классификатор для поиска ближайших точек (изображений) в пространстве фич
print("Apply kNN with "+'metrics='+metric+' and neighbors='+str(n_neighbors))
knn = NearestNeighbors(n_neighbors=n_neighbors,metric=metric, algorithm='brute').fit(pca_features)
# определяем центр в пространстве фич, относительно которого будут нахдиться похожие изображения
pca_features_centroid = np.mean(pca_features, axis = 0)
# находим ближейшие от центра
distances, indices = knn.kneighbors(pca_features_centroid.reshape(-1,pca_comp))
print("Copying files to .\out")
thumbs = []
# копируем в ./out 
for i in indices[0]:
    full_file_name = os.path.join(input_dir, listing[i])
    img = image.load_img(full_file_name)
    thumbs.append(img)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, output_dir)
concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
matplotlib.pyplot.figure(figsize = (16,12))
imshow(concat_image)
matplotlib.pyplot.title("similar images")
