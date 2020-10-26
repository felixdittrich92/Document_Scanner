"""
This file generates the data for the neural network and contains the data class with utils
"""
import os
import re
from tqdm import tqdm

import pandas as pd
import numpy as np 

from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.helper import extract_text, check_language, clean_str, preprocess_texts

FOLDER_PATH = '/home/felix/Desktop/MachineLearning'
FILE_PATH = '/home/felix/Desktop/Document_Scanner/text_classifier/data'

def __unzip_labels(label):
    """unzips a string with more than one word

    Parameters
    ----------
    label : tuple
        tuple or list of the text labels from the folder name

    Returns
    -------
    list
        splited labels as list of strings
    """
    labels = re.split('(?=[A-Z])', label)
    label = [label.lower() for label in labels]
    return label

def __create_parquet(path=FOLDER_PATH, prep_texts=True, save=True):
    """creates a german and english dataframe from the folder/documents

    Parameters
    ----------
    path : str , optional
        path to the root folder from documents and subfolder, by default FOLDER_PATH
    prep_texts : bool, optional
        preprocessing the texts before continue, by default True
    save : bool, optional
        save the dataframes as .parquet file, by default True

    Returns
    -------
    tuple of dataframes
        the generated german and english dataframe from folder/documents
    """
    
    german_labels = list()
    german_texts = list()

    english_labels = list()
    english_texts = list()

    error_files = list()

                #dirs
    for root, _, files in os.walk(path):
        for file in tqdm(files):
            try:
                file_path = os.path.abspath(os.path.join(root, file)) 
                label = os.path.basename(root) 
                label = __unzip_labels(label)
                text = str(extract_text(file_path))
                language = check_language(text)
                if prep_texts:
                    text = preprocess_texts(text, language, filter_stop_words=True)
                if language == 'de':
                    german_labels.append(label) 
                    german_texts.append(text)
                elif language == 'en':
                    english_labels.append(label) 
                    english_texts.append(text)
                else:
                    print(f'can not detect de or en in {file}')
                    error_files.append(file)
            except:
                print(f'{file} occurs an error')
                error_files.append(file)

    print(f'unexcepted files : {error_files}')
            
    # german dataframe
    df_german = pd.DataFrame(columns=['label', 'text'])
    df_german['label'] = german_labels
    df_german['text'] = german_texts

    # english dataframe
    df_english = pd.DataFrame(columns=['label', 'text'])
    df_english['label'] = english_labels
    df_english['text'] = english_texts
    if save:
        df_german.to_parquet(os.path.join(FILE_PATH, 'german.parquet'), index=False)
        df_english.to_parquet(os.path.join(FILE_PATH, 'english.parquet'), index=False)

    print('--------------------------FILES CREATED--------------------------')

    return df_german, df_english


def load_data(language, creating_parquet):
    """loads the dataframe if exist else it will created

    Parameters
    ----------
    language : str
        load german or english dataframe
    creating_parquet : bool
        if used it creates new dataframes from

    Returns
    -------
    dataframe
        the loaded or created dataframe
    """
    if language == 'de':
        parquet_file = os.path.join(FILE_PATH, "german.parquet")    
    elif language == 'en':
        parquet_file = os.path.join(FILE_PATH, "english.parquet")

    if not os.path.isfile(parquet_file) or creating_parquet:
        df_german, df_english = __create_parquet()
        if language == 'de':
            df = df_german
        elif language == 'en':
            df = df_english
    else:
        df = pd.read_parquet(parquet_file)

    return df

class Data():
    """Data class to train a neural network
    """
    def __init__(self, language, creating_parquet):
        """Creates a data instance with the given arguments

        Parameters
        ----------
        language : str
            load german or english dataframe
        creating_parquet : bool
            if used it creates new dataframes from
        """
        self.dataframe = load_data(language=language, creating_parquet=creating_parquet)
        self.texts = self.dataframe['text']
        self.labels = self.dataframe['label']
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_labels(self):
        """preprocess the labels from dataframe for neural network usage
        """
        # initialize MultiLabelBinarizer 
        mlb = MultiLabelBinarizer() 

        # transform the label column to a series of columns with binary values
        binary_labels = pd.DataFrame(mlb.fit_transform(self.dataframe['label']), columns=mlb.classes_) 
        binary_labels = binary_labels.sort_index(axis=1)

        # bring data frames together
        self.dataframe = self.dataframe.merge(binary_labels, how='inner', left_index=True, right_index=True)
        self.labels = self.dataframe[mlb.classes_]

    def preprocess_texts(self, num_words, maxlen):
        """preprocess the texts from dataframe for neural network usage

        Parameters
        ----------
        num_words : int
            the maximum number of words to keep, based on word frequency
        maxlen : int
            the maximum length of all sequences
        """
        tokenizer = Tokenizer(num_words=num_words, split=' ')
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)
        self.texts = pad_sequences(sequences, maxlen=maxlen)

    def split_data(self, test_size=0.25):
        """splits the data into train and test size

        Parameters
        ----------
        test_size : float, optional
            split size, by default 0.25
        """
        self.x_train, self.x_test, self.y_train, self.y_test =\
        train_test_split(self.texts, self.labels, test_size=test_size) 
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

    def get_num_classes(self):
        """counts the number of various labels

        Returns
        -------
        int
            number of various labels
        """
        raw_labels = self.dataframe['label']

        count  = pd.Series(raw_labels.map(Counter).sum())
        num_classes = len(set(count.index))
        return num_classes

    def get_train_set(self):
        return self.x_train, self.y_train
    
    def get_test_set(self):
        return self.x_test, self.y_test
