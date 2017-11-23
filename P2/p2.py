import cv2
import numpy as np
np.random.seed(0) # fix random seed for reproducibility
import os
import pandas as pd
import keras
from keras import backend as K
from keras import applications, optimizers
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

os.chdir('/Users/hokming/Desktop/MSBD6000B/Project2/data/')

# read train, val and test data set
train = pd.read_csv('train.txt', header=None, sep=' ')
val = pd.read_csv('val.txt', header=None, sep=' ')
test = pd.read_csv('test.txt', header=None, sep=' ')

# set image dimensions before loading images
img_width, img_height = 64, 64 # 224, 224 OR 299, 299

# load image function
def load_images(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    return resized

X_train = []
X_val = []
X_test = []
for row in train.ix[:,0]:
    image = load_images(row)
    X_train.append(image)
for row in val.ix[:,0]:
    image = load_images(row)
    X_val.append(image)
for row in test.ix[:,0]:
    image = load_images(row)
    X_test.append(image)

# change to numpy array
X_train = np.array(X_train)
y_train = pd.get_dummies(train.ix[:, 1])
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = pd.get_dummies(val.ix[:, 1])
y_val = np.array(y_val)

X_test = np.array(X_test)

# generate training images
datagen = ImageDataGenerator(
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')
datagen.fit(X_train)


# set image size for self-made CNN
IMG_CHANNELS = 3
IMG_ROWS = 64
IMG_COLS = 64

# constant
BATCH_SIZE = 256
NB_EPOCH = 50
NB_CLASSES = 5
VERBOSE = 1
VALIDATION_SPLIT = 0.0
OPTIM = RMSprop()

# chagne to float and normalization
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /= 255
X_test /= 255

K.set_image_dim_ordering("tf")


# self-made CNN network
model = Sequential()

model.add(Conv2D(32, (2, 2), padding='same',
                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(64, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer=OPTIM,
              metrics=['accuracy'])

# train the self-made CNN
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    samples_per_epoch=X_train.shape[0],
                    epochs=NB_EPOCH, verbose=VERBOSE)

# validate the self-made model and print the test score and accuracy
print('Testing...')
score = model.evaluate(X_val, y_val,
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])  # 0.8927271355

####################################################################################################################

# re-set image size
img_rows, img_cols, img_channel = 224, 224, 3 # OR 229, 229, 3

# set the base model to VGG16, VGG19, ResNet50, InceptionV3 and Xception respectively
base_model = applications.Xception(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

# add one more layer after the base model
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(NB_CLASSES, activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()

# set batch size and epochs for data generator
batch_size = 32
epochs = 20

# generate training images
train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(X_train)

# train the model
history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=X_train.shape[0] // batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint('Xception-transferlearning.model', monitor='val_acc', save_best_only=True)])

# validate the model and print the test score and accuracy
print('Testing...')
score = model.evaluate(X_val, y_val,
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

# use the best model to predict the test data set
predictions = model.predict(X_test)

# change it back to 5 classes
y_classes = predictions.argmax(axis=-1)

# output the predicted test result to csv
np.savetxt("/Users/hokming/Desktop/MSBD6000B/Project2/project2_20386486.txt", result, delimiter=",")