"""Script to train the BertTransformer model
"""
import ktrain
from ktrain import text

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Fix CuDnn problem
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

from sklearn.model_selection import train_test_split

from Dataset import Data

data = Data(language='en', creating_parquet=False) # 'de'
classes = data.get_num_classes()
data.preprocess_labels()
dataframe = data.dataframe
labels = list(dataframe.columns.values)
labels = [label for label in labels if label not in ['text', 'label']]


(x_train, y_train), (x_test, y_test), preproc =\
     text.texts_from_df(
                        dataframe, 
                        text_column='text',
                        label_columns=labels, 
                        maxlen=200, 
                        max_features=3500, 
                        preprocess_mode='bert', 
                        verbose=1
                        )

model = text.text_classifier(
                             'bert', 
                             (x_train, y_train), 
                             preproc=preproc, 
                             multilabel=True, 
                             metrics=['accuracy'], 
                             verbose=1
                             )
learner = ktrain.get_learner(
                             model, 
                             train_data=(x_train, y_train),
                             val_data=(x_test, y_test),
                             batch_size=8
                             )

model_log_dir='/home/felix/Desktop/Document_Scanner/text_classifier/logs/english_transformer_model'
tb_callback = TensorBoard(
    log_dir=model_log_dir,
    histogram_freq=1,  
    write_graph=True)


learner.lr_find(show_plot=True)
learner.autofit(
                lr=1e-4, 
                epochs=150, 
                early_stopping=5, 
                reduce_on_plateau=3,
                reduce_factor=0.95,
                monitor='val_loss', 
                callbacks=[tb_callback], 
                verbose=1
                )

predictor = ktrain.get_predictor(model=learner.model, preproc=preproc, batch_size=8)
predictor.save('/home/felix/Desktop/Document_Scanner/text_classifier/models/english_transformer_model')
