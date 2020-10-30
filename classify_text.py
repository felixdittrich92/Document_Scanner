"""Script to predict the labels of a Text/Document and optional generate a Metafile (json)
"""
import os
import io
import sys
import gc
import argparse
import warnings
from shutil import unpack_archive
import tempfile

import json
from datetime import datetime, date

from googletrans import Translator

import numpy as np
import tensorflow as tf
import ktrain

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU Usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable TF terminal output

from text_classifier.utils.helper import extract_text, preprocess_texts, check_language, clean_str

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

def main():
  parser = argparse.ArgumentParser(description='classify the documents - classes: usa, corona, politik, boxen, sport, rezept, china, python, typisierung, italien')
  parser.add_argument("--path", help='path to a single document or a folder with documents', type=str, required=True)
  parser.add_argument("--meta", help='creates a JSON-File with the predicted labels and the text language', type=__check_for_boolean_value, required=True, default=False)
  parser.add_argument("--summarize", help='summarize the text and append to the JSON-File (Optional if --meta is True)',type=__check_for_boolean_value, required=False, default=False)
  parser.add_argument("--custom", help='predict with your custom trained models (use pretrained if not exist)',type=__check_for_boolean_value, required=False)
  
  args = parser.parse_args()
  path = args.path
  meta = args.meta
  custom = args.custom
  summarize = False
  if meta:
    summarize = args.summarize

  if custom and os.path.exists('text_classifier/custom_data/model/german_model') and any(os.scandir('text_classifier/custom_data/model/german_model')):
    print("\ngerman model - using custom model")
    GERMAN_MODEL_DIR = "text_classifier/custom_data/model/german_model/"
    unpacked_german_predictor = None
  else:
    print("\ngerman model - using pretrained model")
    t = tempfile.TemporaryDirectory()
    with tempfile.TemporaryDirectory() as t:
      print("unpack model running...")
      unpack_archive("text_classifier/models/german_transformer_model/tf_model.zip", t, format = 'zip') # or use: german_distilbert_model  -> faster but not as accurate
      GERMAN_MODEL_DIR = str(t) + '/' 
      unpacked_german_predictor = ktrain.load_predictor(GERMAN_MODEL_DIR)

  
  if custom and os.path.exists('text_classifier/custom_data/model/english_model') and any(os.scandir('text_classifier/custom_data/model/english_model')):
    print("english model - using custom model")
    ENGLISH_MODEL_DIR = "text_classifier/custom_data/model/english_model/"
    unpacked_english_predictor = None
  else:
    print("english model - using pretrained model")
    t = tempfile.TemporaryDirectory()
    with tempfile.TemporaryDirectory() as t:
      print("unpack model running...")
      unpack_archive("text_classifier/models/english_transformer_model/tf_model.zip", t, format = 'zip') # or use: english_distilbert_model  -> faster but not as accurate
      ENGLISH_MODEL_DIR = str(t) + '/' 
      unpacked_english_predictor = ktrain.load_predictor(ENGLISH_MODEL_DIR)
  
  if custom and os.path.exists('text_classifier/custom_data/lda/german_model') and any(os.scandir('text_classifier/custom_data/lda/german_model')):
    print("german lda - using custom lda")
    GERMAN_LDA = "text_classifier/custom_data/lda/german_model/"
  else:
    print("german lda - using pretrained lda")
    GERMAN_LDA = "text_classifier/models/german_LDA/"
  
  if custom and os.path.exists('text_classifier/custom_data/lda/english_model') and any(os.scandir('text_classifier/custom_data/lda/english_model')):
    print("english lda - using custom lda")
    ENGLISH_LDA = "text_classifier/custom_data/lda/english_model/"
  else:
    print("english lda - using pretrained lda")
    ENGLISH_LDA = "text_classifier/models/english_LDA/"
  
  print("\nloading the models takes some time")

  try:
    if unpacked_german_predictor is not None:
      german_predictor = unpacked_german_predictor
    else:
      german_predictor = ktrain.load_predictor(GERMAN_MODEL_DIR)
  except:
    print("loading german model causes an error - can not find model")
    sys.exit()
  try:
    german_lda = ktrain.text.load_topic_model(GERMAN_LDA)
  except:
    print("loading german lda causes an error - can not find model")
    sys.exit()
  try:
    if unpacked_english_predictor is not None:
      english_predictor = unpacked_english_predictor
    else:
      english_predictor = ktrain.load_predictor(ENGLISH_MODEL_DIR)
  except:
    print("loading english model causes an error - can not find model")
    sys.exit()
  try:
    english_lda = ktrain.text.load_topic_model(ENGLISH_LDA)
  except:
    print("loading english lda causes an error - can not find model")
    sys.exit()
  
  check = input("Continue and start predicting? [y/n]")
  if check.lower() != 'y':
    sys.exit()
        

  predict(path, meta, summarize, german_predictor, german_lda, english_predictor, english_lda)

def predict(path, meta, summarize, german_predictor, german_lda, english_predictor, english_lda):
  """predicts the labels of an documents text and generates a metafile if given 

  Parameters
  ----------
  path : str
      path to the file or folder with documents
  meta : bool
      if true a json metafile will be created
  summarize : bool
      if true the metafile extends a text summarization of the actual text
  german_predictor : object
      load saved predictor 
  german_lda : object
      load saved topic model 
  english_predictor : object
      load saved predictor
  english_lda : object
      load saved topic model 
  """
  docs = list()

  if os.path.isfile(path):
    docs.append(path)
  elif os.path.isdir(path):
    subfolder = [f.path for f in os.scandir(path) if f.is_dir()]
    if subfolder:
      warnings.warn("Currently only single files and folders without subfolders supported")
      sys.exit()
    doc_names = [f for f in os.listdir(path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'pdf', 'txt', 'html'))]
    for doc_name in doc_names:
      doc_path = os.path.join(path, doc_name)
      docs.append(doc_path)

  num = 1

  for doc_path in docs:
    topic_list = list()
    classes = list()
    file_path, doc_name = os.path.split(doc_path)
    try:
      text = extract_text(doc_path)
      lang = check_language(text)
      print('--------------------------START PREDICTION: %d / %d --------------------------' % (num, len(docs)))
      if lang == 'de':
        predictor = german_predictor
        topic_model = german_lda
      elif lang == 'en':
        predictor = english_predictor
        topic_model = english_lda
      else:
        print("Document name : %s has no recognizable language only german and english language supported" % (doc_name))
        num += 1
        continue 
      print("Document name : %s\n " % (doc_name))
      prediction = predictor.predict(text)
      for preds in prediction:
        pred_class, proba = preds
        if proba > 0.5:
          # translate labels if english text
          if lang == 'en':
            try:
              translator = Translator()
              pred_class = translator.translate(pred_class, dest='en').text
            except:
              print("is already translated")
          classes.append(pred_class)
        print("Class : %s\nProbability : %f\n" % (pred_class, proba))
      doc_text = [text]
      topic_model.build(doc_text, threshold=0.4)
      topics = topic_model.topics[ np.argmax(topic_model.predict(doc_text))]
      topic_list.append(topics)
      gc.collect()
      num += 1
    except: 
      print("Document name : %s cannot be read or has no text" % (doc_name))
      continue
    
    if meta and lang is not None:
      print('--------------------------CREATE METAFILE--------------------------')
      dict_keys = ["Filename", "Date", "Time", "Language", "Labels", "Keywords"] 
      meta_dict = dict.fromkeys(dict_keys, None) 
      meta_dict["Filename"] = doc_name
      meta_dict["Date"] = date.today().strftime("%b-%d-%Y")
      meta_dict["Time"] = datetime.now().strftime("%H:%M:%S")
      meta_dict["Language"] = lang
      meta_dict["Labels"] = classes 
      meta_dict["Keywords"] = topic_list
      if summarize:
        ts = ktrain.text.TransformerSummarizer()
        text = clean_str(text)
        meta_dict["Summary"] = ts.summarize(text)
      __create_meta(file_path=file_path, doc_name=doc_name, data=meta_dict)

def __create_meta(file_path, doc_name, data):
  """saves the dictonary as json file in the folder of the document

  Parameters
  ----------
  file_path : str
      path to the current file
  doc_name : str
      name of the current document
  data : dict
      the generated dictonary
  """
  doc_name = doc_name.split('.')[0]
  filename = '/'.join((file_path, doc_name))
  with open('.'.join((filename, 'json.meta')), 'w+', errors='ignore') as fp:
    json.dump(data, fp)

if __name__ == "__main__":
  main()
