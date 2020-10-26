"""Convolutional Neural Network Implementation
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

from Dataset import Data


data = Data(extracting_images=True)
data.data_augmentation(augment_size=1200)
x_train_splitted, x_val, y_train_splitted, y_val = data.get_splitted_train_validation_set()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()
num_classes = data.num_classes

# Define the CNN
def model_cnn(optimizer, learning_rate, dropout_rate,
              filter_block1, kernel_size_block1, 
              filter_block2, kernel_size_block2, 
              kernel_size_block3, filter_block3, 
              dense_layer_size, kernel_initializer, 
              bias_initializer, activation_str):
    """Creates the CNN model

    Parameters
    ----------
    optimizer : object
        forms the weights depending on the loss function
    learning_rate : float
        hyperparameter that controls how strongly the weights of the network 
        are adjusted in relation to the loss gradient
    dropout_rate: float
        percentage of ignored neurons 
    filter_block : int
        dimensionality of the output space
    kernel_size_block : int
        size of the convolution window (3,3)
    dense_layer_size : int
        dimensionality of the output space
    kernel_initializer : str
        Initializer for the kernel weights matrix
    bias_initializer : str
        Initializer for the bias vector
    activation_str : str
        determine the output

    Returns
    -------
    Model object
        trained model
    """
    # Input
    input_img = Input(shape=x_train.shape[1:])
    # Conv Block 1
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same',kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same',kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_str)(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    # Conv Block 2
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same',kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same',kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_str)(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    # Conv Block 3
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same',kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same',kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_str)(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    # Dense Part
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation_str)(x)
    x = Dense(units=dense_layer_size)(x)
    x = Activation(activation_str)(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("sigmoid")(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model

# Global params
epochs = 400
batch_size = 8 

params = {
    "optimizer": Adam,
    "learning_rate": 0.001,
    "dropout_rate": 0.2,
    "filter_block1": 16,
    "kernel_size_block1": 3,
    "filter_block2": 16,
    "kernel_size_block2": 3,
    "filter_block3": 32,
    "kernel_size_block3": 3,
    "dense_layer_size": 128,
    "kernel_initializer": 'GlorotUniform',
    "bias_initializer": 'zeros',
    "activation_str": 'relu',
}

# Save logs
model_log_dir = "../Document_Scanner/evaluator/logs/model_test"

model = model_cnn(**params)

plateau_callback = ReduceLROnPlateau(
    monitor='val_accuracy', 
    factor=0.95,
    patience=2,
    verbose=1,
    min_lr=1e-5)

es_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    verbose=1,
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
    validation_data=(x_test, y_test)) 

score = model.evaluate(
    x_test, 
    y_test, 
    verbose=0, 
    batch_size=batch_size)

print(f"test loss : {score[0]} test acc : {score[1]}")

# Save model -> without .h5 it will save as .pb graph
model.save("../Document_Scanner/evaluator/models/model_test/doc_model.h5", save_format='h5')