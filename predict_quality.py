"""Script to predict the quality of an document image
"""
import os
import sys
import argparse
import warnings

import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU Usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable TF terminal output

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

CLASS_IDXS = ["not good", "good"]
MODEl_DIR = "evaluator/models/cnn_model/doc_model.h5"

def main():
  parser = argparse.ArgumentParser(description='check the quality of Images for OCR processing')
  parser.add_argument("--path", help='path to a single image or a folder with images', type=str, required=True)
  args = parser.parse_args()

  path = args.path
  predict(path)

def predict(path):
  """predicts the quality of an image

  Parameters
  ----------
  path : str
      path to the file or folder
  """
  model = load_model(MODEl_DIR)

  images = list()
  if os.path.isfile(path):
    images.append(path)
  elif os.path.isdir(path):
    subfolder = [f.path for f in os.scandir(path) if f.is_dir()]
    if subfolder:
      warnings.warn("Currently only single files and folders without subfolders supported")
      sys.exit()
    image_names = [f for f in os.listdir(path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    for image_name in image_names:
      image_path = os.path.join(path, image_name)
      images.append(image_path)
    
  for image_path in images:
    _, image_name = os.path.split(image_path)
    try:
      image = __load_and_preprocess_custom_image(image_path)
      y_pred = model.predict(np.expand_dims(image, axis=0), verbose=1)[0] 
      y_pred_class = np.argmax(y_pred)
      y_pred_prob = y_pred[y_pred_class]*100 
      score = __calculate_score(y_pred_class, y_pred_prob)
      print("Image name : %s\n Predicted class : %s\n Score : %f" % (image_name, CLASS_IDXS[y_pred_class], score))
    except:
      print("the file %s cannot found or cannot be read" % image_name)

def __calculate_score(y_pred_class, y_pred_prob):
  """calculates the predicted values between zero and one

  Parameters
  ----------
  y_pred_class : array
      predicted class
  y_pred_prob : float
      predicted accuracy

  Returns
  -------
  float
      calculated score between 0 and 1
  """
  if y_pred_class == 0:
    MAX = 0.5
    scaled_percentage = (y_pred_prob * MAX) / 100
    return MAX - scaled_percentage
  else:
    MAX = 1
    scaled_percentage = (y_pred_prob * MAX) / 100
    return scaled_percentage

def __load_and_preprocess_custom_image(image_path):
  """loads and preprocess the image

  Parameters
  ----------
  image_path : str
      path to the local image

  Returns
  -------
  array
      preprocessed image
  """
  img = load_img(image_path, color_mode = 'grayscale', target_size = (700, 700))
  img = img_to_array(img).astype('float32')/255
  return img


if __name__ == "__main__":
  main()

