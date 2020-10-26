"""Script to train the Multilabel classifier on custom data
"""
import os
import gc
import sys
import argparse

import pandas as pd

import ktrain
from ktrain import text

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MultiLabelBinarizer

# Fix CuDnn problem
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

from Dataset import Data, __create_parquet

def main():
    parser = argparse.ArgumentParser(description='train your custom multilabel classifier')
    parser.add_argument("--path", help='path to the folder with documents', type=str, required=True)
    parser.add_argument("--save_dataset", help='save the dataset for renewed training (bool)', type=__check_for_boolean_value, required=False, default=True)
    parser.add_argument("--maxlen", help='each document can be of most  maxlen  words', type=int, required=False, default=30)
    parser.add_argument("--max_features", help='max num of words to consider in vocabulary', type=int, required=False, default=3500)
    parser.add_argument("--batch_size", help='size of parallel trained data stack', type=int, required=False, default=8)
    parser.add_argument("--epochs", help='size of training iterations', type=int, required=False, default=40)
    parser.add_argument("--lda_n_features", help='number of extracted topics', type=int, required=False, default=150)
    parser.add_argument("--lda_threshold", help='probabilities below this value are filtered out', type=float, required=False, default=0.2)
    args = parser.parse_args()

    path = args.path
    save_dataset = args.save_dataset
    maxlen = args.maxlen

    max_features = args.max_features
    batch_size = args.batch_size
    epochs = args.epochs
    n_features = args.lda_n_features
    threshold = args.lda_threshold

    check = input("To train on your own data you need a Nvidia GPU >= 8 GB VRAM (recommended) and installed the actual driver ! ready ? [y/n]")
    if check.lower() != 'y':
        sys.exit()
        
    if not os.path.exists('text_classifier/custom_data'):
        os.makedirs('text_classifier/custom_data/logs/german_model')
        os.makedirs('text_classifier/custom_data/logs/english_model')
        os.makedirs('text_classifier/custom_data/model/german_model')
        os.makedirs('text_classifier/custom_data/model/english_model')
        os.makedirs('text_classifier/custom_data/lda/german_model')
        os.makedirs('text_classifier/custom_data/lda/english_model')
        df_german, df_english = __create_parquet(path=path, save=False) # do not change
    else:
        df_german = pd.read_parquet('text_classifier/custom_data/german.parquet')
        df_english = pd.read_parquet('text_classifier/custom_data/english.parquet')

    if save_dataset:
        df_german.to_parquet('text_classifier/custom_data/german.parquet', index=False)
        df_english.to_parquet('text_classifier/custom_data/english.parquet', index=False)

    gc.collect()
    if not df_german.empty:
        df, labels = preprocess_labels(dataframe=df_german)
        train_classifier(dataframe=df, labels=labels, maxlen=maxlen, max_features=max_features, batch_size=batch_size, epochs=epochs, model_save_dir='text_classifier/custom_data/model/german_model', model_log_dir='text_classifier/custom_data/logs/german_model')
        train_lda(data=df_german, n_features=n_features, threshold=threshold, save_dir='text_classifier/custom_data/lda/german_model/')
    else:
        print("Can not found german dataset")

    if not df_english.empty:
        df, labels = preprocess_labels(dataframe=df_english)
        train_classifier(dataframe=df, labels=labels, maxlen=maxlen, max_features=max_features, batch_size=batch_size, epochs=epochs, model_save_dir='text_classifier/custom_data/model/english_model', model_log_dir='text_classifier/custom_data/logs/english_model')
        train_lda(data=df_english, n_features=n_features, threshold=threshold, save_dir='text_classifier/custom_data/lda/english_model/')
    else:
        print("Can not found english dataset")
    
    print("_________________TRAINING FINISHED___________________")
    print("NOTE: now you can use classify_text.py with --custom argument")
    print("NOTE: if you have only english or german model the other will be replaced with pretrained")
    print("NOTE: Closing all jobs -- this takes a moment")


def preprocess_labels(dataframe):
    """

    Parameters
    ----------
    dataframe : dataframe
        the generated dataframe from data folder

    Returns
    -------
    tuple
        the processed dataframe and a list with label strings
    """
    mlb = MultiLabelBinarizer() 

    # transform the label column to a series of columns with binary values
    binary_labels = pd.DataFrame(mlb.fit_transform(dataframe['label']), columns=mlb.classes_) 
    binary_labels = binary_labels.sort_index(axis=1)

    # bring data frames together
    dataframe = dataframe.merge(binary_labels, how='inner', left_index=True, right_index=True)
    labels = dataframe[mlb.classes_]
    labels = list(dataframe.columns.values)
    labels = [label for label in labels if label not in ['text', 'label']]
    return dataframe, labels


def train_classifier(dataframe, labels, maxlen, max_features, batch_size, epochs, model_save_dir, model_log_dir):
    """train the multilabel classifier

    Parameters
    ----------
    dataframe : dataframe
        the generated dataframe from data folder
    labels : str
        the extracted labels 
    maxlen : int
        each document can be of most  maxlen  words
    max_features : int
        max num of words to consider in vocabulary
    batch_size : int
        size of parallel trained data stack
    epochs : int
        size of training iterations
    model_save_dir : str
        path to save the model
    model_log_dir : str
        path to the tracked logs for tensorboard usage
    """
    (x_train, y_train), (x_test, y_test), preproc =\
        text.texts_from_df(
                            dataframe, 
                            text_column='text',
                            label_columns=labels, 
                            maxlen=maxlen, 
                            max_features=max_features, 
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
                                batch_size=batch_size
                                )

    tb_callback = TensorBoard(
        log_dir=model_log_dir,
        histogram_freq=1,  
        write_graph=True)


    learner.lr_find(show_plot=True)
    learner.autofit(
                    lr=1e-4, 
                    epochs=epochs, 
                    early_stopping=5, 
                    reduce_on_plateau=3,
                    reduce_factor=0.95,
                    monitor='val_loss', 
                    callbacks=[tb_callback], 
                    verbose=1
                    )

    predictor = ktrain.get_predictor(model=learner.model, preproc=preproc, batch_size=batch_size)
    predictor.save(model_save_dir)

def train_lda(data, n_features, threshold, save_dir):
    """train the latent dirichlet allocation

    Parameters
    ----------
    data : dataframe
        the generated dataframe from data folder
    n_features : int
        number of extracted topics
    threshold : float
        probabilities below this value are filtered out
    save_dir : str
        path to the model save folder
    """
    tm = ktrain.text.get_topic_model(data['text'], n_features=n_features)
    tm.print_topics()
    tm.build(data['text'], threshold=threshold)
    tm.print_topics(show_counts=True)
    tm.save(save_dir)

def __check_for_boolean_value(val):

  """argparse helper function 

  Parameters
  ----------
  val : str
      return from parser

  Returns
  -------
  bool
      true value
  """
  if val.lower() == "true":
    return True
  else:
    return False

if __name__ == "__main__":
  main()

