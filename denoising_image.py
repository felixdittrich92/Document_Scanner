"""Skript to denoise a image
"""
import os
import argparse

import cv2

import numpy as np
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU Usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable TF terminal output

# Fix CuDnn problem
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import *

MODEl_DIR = "denoiser/models/auto_model/auto_model.h5"

def main():
  parser = argparse.ArgumentParser(description='denoises an document image')
  parser.add_argument("--input", help='path to a single image', type=str, required=True)
  parser.add_argument("--output", help='name of the output image (example: out.png / out.jpg) ', type=str, required=True)
  args = parser.parse_args()

  input_path = args.input
  output = args.output
  denoising_image(input_path, output)

def denoising_image(input_path, output):
  """

  Parameters
  ----------
  path : str
      full path to the image
  out : str
      the name of the output file 
  """
  model = load_model(MODEl_DIR)

  try:
    org_img = load_img(input_path, color_mode = 'grayscale')
    org_img = img_to_array(org_img)
    img = org_img.astype('float32')
    img = np.expand_dims(org_img, axis=0)
    y_pred = np.squeeze(model.predict(img, verbose=1))
    img = cv2.convertScaleAbs(y_pred, alpha=(255.0))
    cv2.imwrite(output, img)
  except:
    _, image_name = os.path.split(input_path)
    print("the file %s is to large" % image_name)

if __name__ == "__main__":
  main()
