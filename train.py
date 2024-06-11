import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_directory = r'C:\Users\RITIK\Downloads\brain_mri\dataset'

no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))

datasets = []
label = []
input_size=64

# Process images with no tumors
for image_name in no_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image_path = os.path.join(image_directory, 'no', image_name)
        image = cv2.imread(image_path)
        
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((64, 64))  # input size = 64
            datasets.append(np.array(image))
            label.append(0)
        else:
            print(f"Failed to read {image_path}")

# Process images with tumors
for image_name in yes_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image_path = os.path.join(image_directory, 'yes', image_name)
        image = cv2.imread(image_path)
        
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((64, 64))  # input_size=64
            datasets.append(np.array(image))
            label.append(1)
        else:
            print(f"Failed to read {image_path}")

# Convert lists to numpy arrays for further processing
datasets = np.array(datasets)
label = np.array(label)

print("Dataset and labels prepared.")
print(len(datasets))
print(len(label))

x_train,x_test,y_train,y_test=train_test_split(datasets,label,test_size=0.2,random_state=0)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


# Normalise train and test data

x_train= normalize(x_train,1)
x_test= normalize(x_test,1)

# # if we use categorial_cross entropy
# y_train= to_categorical(y_train,num_classes=2)
# y_test= to_categorical(y_test,num_classes=2)

# Model Buildiong
model = Sequential()

model.add(Conv2D(32, (3,3),input_shape=(input_size,input_size,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, (3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
# model.add(Dense(2))  # if we use categorial_crossentropy the value should be 2 
model.add(Activation('sigmoid'))
# model.add(Activation('softmax'))


# Binary Cross Entropy = 1 ,sigmoid
# categorial cross Entropy = 2 , softmax

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test,y_test),shuffle=False)

model.save('BrainTumor10E.h5')
#model.save('BrainTumor10ECategorical.h5')





