
#preprocess

import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, Cropping2D
from keras.layers import Dense, Dropout, Flatten, ELU
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

batch_size = 256
nb_epoch = 1


## 1. Prepare and create generator

# Save filepaths of images to `samples` to load into generator
samples = []

def add_to_samples(csv_filepath, samples):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

pname = '/Users/HARAN/miniconda3/envs/carnd-term1/data/veh_trn_data_p3/'

#samples -->list of list
samples = add_to_samples(pname+'driving_log.csv', samples)

#samples = add_to_samples('data-recovery-annie/driving_log.csv', samples)  # header already removed


# data viz
# Remove header
samples = samples[1:]

print("Samples: ", len(samples))

import matplotlib.pyplot as plt

angles = [float(sample[3]) for sample in samples]

fig,ax = plt.subplots(1,2)
ax[0].hist(angles, normed=False, bins=150)
#ax[0].ylabel('Counts')
# plt.ion()
# plt.show()

#1. reduce number entries by 25% in for data with steer angle of 0
idx_rm = []
for idx,angle in enumerate(angles):
    if abs(angle) <= 0.01:
        idx_rm.append(idx)
print("length of angles in b/w idx_rm:", len(idx_rm))

new_idx_rm = []
for item in idx_rm:
    if np.random.rand()>0.25:
        new_idx_rm.append(item)
print("length of new angles in b/w idx_rm:", len(new_idx_rm))
new_angles = np.delete(angles, new_idx_rm)

samples = np.delete(samples, new_idx_rm, axis=0)
samples.tolist()

#viz
ax[1].hist(new_angles, normed=False, bins=150)
#ax[1].ylabel('New Counts')

plt.ion()
plt.show()

# generate equal amount of right and left images
#if np.random.rand() > 0.5:  # 50 percent chance to see the right angle
    # center_image = cv2.flip(center_image, 1)
    # angle = -angle


# Split samples into training and validation sets to reduce overfitting
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


# def preprocessImage(image):
#     # Proportionally get lower half portion of the image
#     nrow, ncol, nchannel = image.shape
#
#     start_row = int(nrow * 0.35)
#     end_row = int(nrow * 0.875)
#
#     # This removes most of the sky and small amount below including the hood
#     new_image = image[start_row:end_row, :]
#
#     # This resizes to 66 x 220 for NVIDIA's model
#     new_image = cv2.resize(new_image, (220, 66), interpolation=cv2.INTER_AREA)
#
#     return new_image

def read_data(batch_size):
    """
    Generator function to load driving logs and input images.
    """
    while 1:
        with open(pname+'driving_log.csv') as driving_log_file:
            driving_log_reader = csv.DictReader(driving_log_file)
            count = 0
            inputs = []
            targets = []
            try:
                for row in driving_log_reader:
                    steering_offset = 0.4

                    centerImage = mpimg.imread(pname+ row['center'].strip())
                    flippedCenterImage = np.fliplr(centerImage)
                    centerSteering = float(row['steering'])

                    leftImage = mpimg.imread(pname+ row['left'].strip())
                    flippedLeftImage = np.fliplr(leftImage)
                    leftSteering = centerSteering + steering_offset

                    rightImage = mpimg.imread(pname+ row['right'].strip())
                    flippedRightImage = np.fliplr(rightImage)
                    rightSteering = centerSteering - steering_offset

                    if count == 0:
                        inputs = np.empty([0, 160, 320, 3], dtype=float)
                        targets = np.empty([0, ], dtype=float)

                    if count < batch_size:
                        inputs = np.append(inputs, np.array([centerImage, flippedCenterImage, leftImage, flippedLeftImage, rightImage, flippedRightImage]), axis=0)
                        targets = np.append(targets, np.array([centerSteering, -centerSteering, leftSteering, -leftSteering, rightSteering, -rightSteering]), axis=0)
                        count += 6
                    else:
                        yield inputs, targets
                        count = 0
            except StopIteration:
                pass


# 1. Preprocessing (incorporated into the model)
# Model adapted from Comma.ai model

def resize_comma(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, (80, 160))


model = Sequential()

# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                     dim_ordering='tf',  # default
                     input_shape=(160, 320, 3)))

# Resize the data
model.add(Lambda(resize_comma))

#import tensorflow as tf
# model.add(Lambda(lambda x: tf.image.resize_images(x, (80,160))))

#model.add(Lambda(lambda x: preprocessImage(x)))

# Normalise the data
model.add(Lambda(lambda x: (x / 127.5) - 1))

#2. define CNN architecture
######################################################################

#NVIDIA Model

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv1'))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv2'))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv3'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv4'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv5'))
model.add(Flatten(name='flatten1'))
model.add(ELU())
model.add(Dense(1164, init='he_normal', name='dense1'))
model.add(ELU())
model.add(Dense(100, init='he_normal', name='dense2'))
model.add(ELU())
model.add(Dense(50, init='he_normal', name='dense3'))
model.add(ELU())
model.add(Dense(10, init='he_normal', name='dense4'))
model.add(ELU())
model.add(Dense(1, init='he_normal', name='dense5'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

print("Model summary:\n", model.summary())

## 4. Train model
batch_size = 1024
model.fit_generator(read_data(batch_size), samples_per_epoch=8036, nb_epoch=1)


# # Save model weights after each epoch
# checkpointer = ModelCheckpoint(filepath=pname+"tmp/v2-weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1,
#                                save_best_only=False)
#
# # Train model using generator
# # model.fit_generator(train_generator,
# #                     samples_per_epoch=len(train_samples),
# #                     validation_data=validation_generator,
# #                     nb_val_samples=len(validation_samples), nb_epoch=nb_epoch,
# #                     callbacks=[checkpointer])
#
#
# model.fit_generator(train_generator,
#                     validation_data=validation_generator,
#                     samples_per_epoch = len(train_samples),
#                     nb_val_samples=len(validation_samples),nb_epoch=nb_epoch)
#
#
#

## 5. Save model

model.save('model.h5')

