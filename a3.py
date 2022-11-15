# from google.colab import drive
# drive.mount("/content/drive")
import numpy as np
import pandas as pd
import os
import random
import matplotlib.image as mping
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow
from keras.preprocessing.image import ImageDataGenerator

# %matplotlib inline


TrainingImagePath="data/Image/train"
TestImagePath="data/Image/test"
ValidationImagePath="data/Image/valid"

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

##

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    TestImagePath,
    target_size = (128,128),
    batch_size=32,
    class_mode="categorical"
)
valid_set = test_datagen.flow_from_directory(
    ValidationImagePath,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

def showImages(class_name):
    random_index = random.choice(list(range(1,49)))
    folder_path = os.path.join(TrainingImagePath, class_name)
    try:
        image_path = os.path.join(folder_path,str(random_index).zfill(3)+".jpg")
        plt.imshow(mping.imread(image_path))
    except:
        image_path = os.path.join(folder_path,str(random_index).zfill(2)+".jpg")
        plt.imshow(mping.imread(image_path))
    plt.title(class_name)
    plt.axis(False)

plt.figure(figsize = (20,20))

for labels,number in training_set.class_indices.items():
    plt.subplot(6,6,number+1)
    showImages(labels)

test_set.class_indices

'''#################### Creating lookup table for all balls #####
#########################'''
# class_indices have the numeric tag for each balls
TrainClasses=training_set.class_indices
# Storing the face and the numeric tag for future reference
ResultMap={}
for ballValue,ballName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[ballValue]=ballName
# Saving the face map for future reference
import pickle
with open(R"E:\Data Sets\Balls Classification\ResultsMap.pkl", 'wb') as f:
    pickle.dump(ResultMap, f, pickle.HIGHEST_PROTOCOL)

print("Mapping of Face and its ID",ResultMap)

# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

classifier= Sequential()
classifier.add(
    Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=(128,128,3), activation='relu')
)
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(OutputNeurons, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=["accuracy"])

###

classifier.summary()

###

import time
# Measuring the time taken by the model to train
StartTime=time.time()
# Starting the model training
model_history=classifier.fit_generator(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=5,
    validation_data=valid_set,
    validation_steps=len(valid_set),
    verbose=1
)

EndTime=time.time()
print("############### Total Time Taken: ", round((EndTime-
StartTime)/60), 'Minutes #############')
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.plot(accuracy, label = "Training accuracy")
plt.plot(val_accuracy, label="Validation accuracy")
plt.legend()
plt.title("Training vs validation accuracy")
plt.subplot(2,2,2)
plt.plot(loss, label = "Training loss")
plt.plot(val_loss, label="Validation loss")
plt.legend()
plt.title("Training vs validation loss")
plt.show()