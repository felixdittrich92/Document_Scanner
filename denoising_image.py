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
  parser.add_argument("--quality", help='describe the quality for the output image (0-100) ', type=int, required=False, default=95)
  parser.add_argument("--res_width", help='the resolution width for the output image (aspect ratio is calculated)', type=int, required=False, default=None)
  args = parser.parse_args()

  input_path = args.input
  output = args.output
  quality = args.quality
  res_width = args.res_width
  denoising_image(input_path, output, quality, res_width)

def denoising_image(input_path, output, quality, res_width):
  """

  Parameters
  ----------
  path : str
      full path to the image
  out : str
      the name of the output file 
  quality : int
      the quality of the output image
  res_width : int 
      the width size for the output image
  """
  model = load_model(MODEl_DIR)
  dim = None

  try:
    org_img = load_img(input_path, color_mode = 'grayscale')
    if res_width is not None:
      (h, w) = (org_img.size[1], org_img.size[0])
      r = res_width / float(w)
      dim = (res_width, int(h * r))
    org_img = img_to_array(org_img)
    img = org_img.astype('float32')
    img = np.expand_dims(org_img, axis=0)
    y_pred = np.squeeze(model.predict(img, verbose=1))
    img = cv2.convertScaleAbs(y_pred, alpha=(255.0))
    if dim:
      img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(output, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
  except:
    _, image_name = os.path.split(input_path)
    print("the file %s is to large" % image_name)

if __name__ == "__main__":
  main()
