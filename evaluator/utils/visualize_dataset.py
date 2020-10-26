"""visualize data from the numpy file
"""

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

img_array = np.load('../Scans/x.npy')
img = array_to_img(img_array[170])

plt.imshow(img, cmap='gray')
plt.show()

label = np.load('../Scans/y.npy')
print(label[170])