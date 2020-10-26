"""Autoencoder Implementation to denoise images
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Train on CPU

import random

import numpy as np
import tensorflow as tf

# Fix CuDnn problem
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from tensorflow.keras.preprocessing.image import *
from sklearn.model_selection import train_test_split

train_images_path = '../denoising/train'
train_cleaned_path = '../denoising/train_cleaned'

train_images = sorted(os.listdir(train_images_path))
train_cleaned = sorted(os.listdir(train_cleaned_path))

x_train = []
y_train = []

for image in train_images:
    img_path = os.path.join(train_images_path, image)
    img = load_img(img_path, color_mode = 'grayscale', target_size = (420, 540))
    img = img_to_array(img).astype('float32')/255
    x_train.append(img)
for image in train_cleaned:
    img_path = os.path.join(train_cleaned_path, image)
    img = load_img(img_path, color_mode = 'grayscale', target_size = (420, 540))
    img = img_to_array(img).astype('float32')/255
    y_train.append(img)
     
x = np.array(x_train)
y = np.array(y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25)

def autoencoder_model(optimizer, learning_rate, 
                      filter_block1, kernel_size_block1, 
                      filter_block2, kernel_size_block2, 
                      filter_block3, kernel_size_block3, 
                      filter_block4, kernel_size_block4, 
                      activation_str, padding):
    """Creates the Autoencoder model

    Parameters
    ----------
    optimizer : object
        forms the weights depending on the loss function
    learning_rate : float
        hyperparameter that controls how strongly the weights of the network 
        are adjusted in relation to the loss gradient
    filter_block : int
        dimensionality of the output space
    kernel_size_block : int
        size of the convolution window (3,3)
    padding: str
        handles how to proceed at the edges of an image
    activation_str : str
        determine the output

    Returns
    -------
    Model object
        trained model
    """
    # Input Tensors - fully conv
    input_img = Input(shape=(None, None, 1))
    # Encoder Part
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding=padding)(input_img) # 420x540x32
    x = Activation('relu')(x)
    x = MaxPooling2D()(x) # 210x270x32
    encoded = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding=padding)(x) # 105x135x32
    # Decoder Part
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding=padding)(encoded) # 210x270x32
    x = Activation('relu')(x)
    x = UpSampling2D()(x) # 420x540x32
    decoded = Conv2D(filters=filter_block4, kernel_size=kernel_size_block4, activation='sigmoid', padding=padding)(x) # 420x540x1

    # Build the model
    autoencoder = Model(inputs=input_img, outputs=decoded)
    opt = optimizer(learning_rate=learning_rate)
    autoencoder.compile(loss="binary_crossentropy", optimizer=opt)
    autoencoder.summary()
    return autoencoder

# Global params
epochs = 2000
batch_size = 8

params = {
    "optimizer": Adam,
    "learning_rate": 0.0001,
    "filter_block1": 32,
    "kernel_size_block1": 4,
    "filter_block2": 32,
    "kernel_size_block2": 3,
    "filter_block3": 32,
    "kernel_size_block3": 3,
    "filter_block4": 1,
    "kernel_size_block4": 4,
    "padding": 'same',
    "activation_str": 'relu'
}

# Save logs
model_log_dir = "../Document_Scanner/denoiser/logs/auto_model"

model = autoencoder_model(**params)

plateau_callback = ReduceLROnPlateau(
    monitor='loss', 
    factor=0.95,
    patience=2,
    verbose=1,
    min_lr=1e-6)

es_callback = EarlyStopping(
    monitor='loss',
    patience=15,
    verbose=1,
    min_delta=1e-4,
    restore_best_weights=True)

tb_callback = TensorBoard(
    log_dir=model_log_dir,
    histogram_freq=1,  
    write_graph=True)

model.fit(
    x=x_train, 
    y=y_train, 
    verbose=1, 
    batch_size=batch_size, 
    epochs=epochs, 
    callbacks=[tb_callback, plateau_callback, es_callback],
    validation_data = (x_valid, y_valid)) 


# Save model -> without .h5 it will save as .pb graph
model.save("../Document_Scanner/denoiser/models/auto_model/auto_model.h5", save_format='h5')