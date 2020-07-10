import csv
import cv2
from scipy import ndimage
# cd CarND-Behavioral-Cloning-P3
# python model.py
# delete images rm -f IMG/*
# copy images cp -R ~/Desktop/IMG /home/workspace/CarND-Behavioral-Cloning-P3/data
# copy log    cp -R ~/Desktop/driving_log.csv /home/workspace/CarND-Behavioral-Cloning-P3/data
# run the model python drive.py model.h5
# record video images python drive.py model.h5 run1
# record video python video.py run1

#########################################################
# Start rearing the driving_log.csv files and creating
# an array with the name of the images to train the model
#############################################################

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
DATA_AUGMENTATION = False
NUMBER_OF_CAMERAS = 3 #1,2,3

########################################################
# Use the name and location of the images to read them
# and create an array with them
#########################################################
print("Reading {0} images according to driving_log.csv...".format(len(lines) -1))
images = []
measurements = []
for line in lines:
    for i in range(NUMBER_OF_CAMERAS):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
    
        #image = cv2.imread(current_path)
    
        # NOTE: cv2.imread will get images in BGR format, while drive.py uses RGB. 
        # One way you could keep the same image formatting is to do 
        # "image = ndimage.imread(current_path)" with "from scipy import ndimage" instead.
        image = ndimage.imread(current_path)    
        images.append(image)
        measurement = float(line[3])
        #are we using 3 cameras?
        correction = 0.7 # this is a parameter to tune
        if(i == 1):            
            measurement = measurement + correction #steering_left
        if(i == 2):            
            measurement = measurement - correction #steering_right
            
        measurements.append(measurement)
    
        if(DATA_AUGMENTATION == True):
            #image_flipped = ndimage.interpolation.rotate(image,180.0,axes=(0,1))
            image_flipped = np.fliplr(image)
            measurement_flipped = -measurement
            images.append(image_flipped)
            measurements.append(measurement_flipped)


print("Number of images found: {0} = measurements {1}".format(len(images), len(measurements)))

#########################################################
import numpy as np

#Keras requires numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

########################################################
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

USE_SIMPLE_MODEL = False
USE_LeNet = False
USE_NVDIA = True

input_shape = (160,320,3)

model = Sequential()
#normalize the images. Set up lambda layer
model.add(Lambda(lambda x: (x /255.0) - 0.5, input_shape = input_shape))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
#50 rows pixels from the top of the image
#20 rows pixels from the bottom of the image
#0 columns of pixels from the left of the image
#0 columns of pixels from the right of the image
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=input_shape))

if (USE_SIMPLE_MODEL == True):
    model.add(Flatten())
    model.add(Dense(1))
    #Regresion network
    model.compile(loss='mse', optimizer='adam')
    # Fit the model
    history = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)

elif (USE_LeNet == True):
    #LeNet
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    #Regresion network
    model.compile(loss='mse', optimizer='adam')
    # Fit the model
    history = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)

elif(USE_NVDIA == True):
    #NVDIA
    #Convolutional feature map 24, Kernel 5x5
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,subsample=(2,2),activation='relu'))
    model.add(Flatten())
    #Neurons
    model.add(Dense(100))
    model.add(Dense(50))    
    model.add(Dense(1))

    #Regresion network
    model.compile(loss='mse', optimizer='adam')
    # Fit the model
    history = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)

    
model.save('model.h5')

########################################################
import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())

###########################
# summarize history for accuracy
fig = plt.figure(figsize=(3, 6))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

fig.savefig('IMG/model_accuracy.png', dpi=fig.dpi)

###########################
# summarize history for loss

fig = plt.figure(figsize=(3, 6))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

exit()