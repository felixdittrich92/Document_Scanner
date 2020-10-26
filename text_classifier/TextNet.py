"""Deprecated - only old test file
"""

import os

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

num_words = 3500
maxlen = 200
embedding_dim = 100
data = Data(language='en', creating_parquet=True)
classes = data.get_num_classes()
data.preprocess_labels()
data.preprocess_texts(num_words=num_words, maxlen=maxlen)
data.split_data(test_size=0.25)
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()

def model_lstm(optimizer, learning_rate,
               num_words, embedding_dim,
               maxlen, num_classes):
    
    # Input
    input_text = Input(shape=x_train.shape[1:])
    # Embedding
    x = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen)(input_text)
    x = LSTM(units=400)(x)
    x = Dense(units=classes)(x)
    output_pred = Activation("softmax")(x)

    model = Model(inputs=input_text, outputs=output_pred)
    opt = optimizer(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model

# Global params
epochs = 100000
batch_size = 256 

params = {
    "optimizer": Adam,
    "learning_rate": 0.0001,
    "num_words": num_words,
    "embedding_dim": embedding_dim,
    "maxlen": maxlen,
    "num_classes": classes
}

# Save logs
model_log_dir = "../Document_Scanner/text_classifier/logs/text_model"

model = model_lstm(**params)

plateau_callback = ReduceLROnPlateau(
    monitor='loss', 
    factor=0.95,
    patience=2,
    verbose=1,
    min_lr=1e-5)

es_callback = EarlyStopping(
    monitor='loss',
    patience=100,
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
    validation_split=0.1) 


# Save model -> without .h5 it will save as .pb graph
#model.save("../Document_Scanner/text_classifier/models/text_model/text_model.h5", save_format='h5')
