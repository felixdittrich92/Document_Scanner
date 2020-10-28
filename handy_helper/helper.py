""" script to prepare a handy image for OCR, Autoencoder, etc.
"""
import argparse

from imutils.perspective import four_point_transform
import cv2
import numpy as np

def __check_for_boolean_value(val):
    if val.lower() == "true":
        return True
    else:
        return False

def main():
  parser = argparse.ArgumentParser(description='prepares a phone image for OCR')
  parser.add_argument("--input", help='path to a single image', type=str, required=True)
  parser.add_argument("--output", help='name of the output image (example: out.png / out.jpg) ', type=str, required=True)
  parser.add_argument("--sharp", help='sharpening the image (example: --sharp=True) else no sharpening', type=__check_for_boolean_value, required=False)
  parser.add_argument("--quality", help='describe the quality for the output image (0-100) ', type=int, required=False, default=95)
  args = parser.parse_args()

  input_path = args.input
  output = args.output
  sharpen = args.sharp
  quality = args.quality
  warped = preprocessing_handy_image(input_path, output, sharpen)
  cv2.imwrite(output, warped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def preprocessing_handy_image(path, output, sharpen=False):
  """

  Parameters
  ----------
  path : str
      full path to the image
  out : str
      the name of the output file 

  Returns
  -------
  object
      processed image
  """
  # load image, grayscale, Gaussian blur, Otsu's threshold
  image = cv2.imread(path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # find contours and sort for largest contour
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  displayCnt = None

  for c in cnts:
      # perform contour approximation
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      if len(approx) == 4:
          displayCnt = approx
          break

  # obtain birds' eye view of image
  image = four_point_transform(image, displayCnt.reshape(4, 2))
  if sharpen:
    # sharpening image
    kernel_filter = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])   

    image = cv2.filter2D(image, -1, kernel_filter)           
  return image

if __name__ == "__main__":
  main()
