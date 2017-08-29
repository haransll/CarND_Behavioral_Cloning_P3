# preprocess

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

augment_const = 6

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

# samples -->list of list
samples = add_to_samples(pname + 'driving_log.csv', samples)

# samples = add_to_samples('data-recovery-annie/driving_log.csv', samples)  # header already removed


# data viz
# Remove header
samples = samples[1:]

print("Samples: ", len(samples))

import matplotlib.pyplot as plt

angles = [float(sample[3]) for sample in samples]

fig, ax = plt.subplots(1, 2)
ax[0].hist(angles, normed=False, bins=150)
# ax[0].ylabel('Counts')
# plt.ion()
# plt.show()

# 1. reduce number entries by 75% in for data with steer angle of 0
idx_rm = []
for idx, angle in enumerate(angles):
    if abs(angle) <= 0.01:
        idx_rm.append(idx)
print("length of angles in b/w idx_rm:", len(idx_rm))

new_idx_rm = []
for item in idx_rm:
    if np.random.rand() > 0.25:
        new_idx_rm.append(item)
print("length of new angles in b/w idx_rm:", len(new_idx_rm))
new_angles = np.delete(angles, new_idx_rm)

samples = np.delete(samples, new_idx_rm, axis=0)
samples.tolist()

# Data viz
ax[1].hist(new_angles, normed=False, bins=150)
# ax[1].ylabel('New Counts')

plt.ion()
plt.show()

# generate equal amount of right and left images
# if np.random.rand() > 0.5:  # 50 percent chance to see the right angle
# center_image = cv2.flip(center_image, 1)
# angle = -angle


# Split samples into training and validation sets to reduce overfitting
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


def preprocessImage(image):
    # Proportionally get lower half portion of the image
    nrow, ncol, nchannel = image.shape

    start_row = int(nrow * 0.35)
    end_row = int(nrow * 0.875)

    # This removes most of the sky and small amount below including the hood
    new_image = image[start_row:end_row, :]

    # This resizes to 66 x 220 for NVIDIA's model
    new_image = cv2.resize(new_image, (220, 66), interpolation=cv2.INTER_AREA)

    return new_image


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates

        shuffle(samples)

        # extract a batch of samples
        steering_offset = 0.5
        step_size = batch_size//augment_const
        for offset in range(0, num_samples, step_size):
            batch_samples = samples[offset:offset + step_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                center_image = mpimg.imread(pname + batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                flipped_center_image = np.fliplr(center_image)
                images.append(flipped_center_image)
                angles.append(-center_angle)

                if abs(center_angle) > 0.01:

                    '''left image'''
                    left_image = mpimg.imread(pname + batch_sample[1].strip())
                    left_angle = center_angle + steering_offset
                    images.append(left_image)
                    angles.append(left_angle)

                    flipped_left_image = np.fliplr(left_image)
                    images.append(flipped_left_image)
                    angles.append(-left_angle)

                    '''right image'''
                    right_image = mpimg.imread(pname + batch_sample[2].strip())
                    right_angle = center_angle - steering_offset
                    images.append(right_image)
                    angles.append(right_angle)

                    flipped_right_image = np.fliplr(right_image)
                    images.append(flipped_right_image)
                    angles.append(-right_angle)


            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)


# 1. Preprocessing (incorporated into the model)
# Model adapted from Comma.ai model

def resize_input_image(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, (80, 160))


model = Sequential()

# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0)),input_shape=(160, 320, 3)))

# Resize the data
model.add(Lambda(resize_input_image))

# import tensorflow as tf
# model.add(Lambda(lambda x: tf.image.resize_images(x, (80,160))))

#model.add(Lambda(preprocessImage))

# Normalise the data
model.add(Lambda(lambda x: (x / 255) - 0.5))

# 2. define CNN architecture
######################################################################


model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())

# Conv layer 2
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())

# Conv layer 3
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())

model.add(Flatten())
#model.add(Dropout(.2))


# Fully connected layer 1
model.add(Dense(512))
model.add(ELU())
model.add(Dropout(.5))


# Fully connected layer 2
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(.5))

# Fully connected layer 3
model.add(Dense(10))
model.add(ELU())
model.add(Dropout(.5))

#FC output layer
model.add(Dense(1))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

print("Model summary:\n", model.summary())

## 4. Train model


# Save model weights after each epoch
# checkpointer = ModelCheckpoint(filepath=pname + "tmp/v2-weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1,
#                                save_best_only=False)
#
# '''Callback:  stop training when no change in validation loss'''
# checkpointer = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=5, mode='auto')


# Train model using generator
# model.fit_generator(train_generator,
#                     samples_per_epoch=len(train_samples),
#                     validation_data=validation_generator,
#                     nb_val_samples=len(validation_samples), nb_epoch=nb_epoch,
#                     callbacks=[checkpointer])


model.fit_generator(train_generator,
                    samples_per_epoch=augment_const*len(train_samples),
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples))

## 5. Save model

model.save('model.h5')

print(model.summary())

print("end program")
