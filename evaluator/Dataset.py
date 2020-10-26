"""
This file generates the data for the neural network and contains the data class with utils
"""
import os

import numpy as np

from sklearn.model_selection import train_test_split

from skimage import transform

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# path to folder with the dataset
FILE_DIR = os.path.abspath("../Scans") 
IMG_WIDTH = 700
IMG_HEIGHT = 700
IMG_DEPTH = 1

WORST_CLASS_IDX = 0
BEST_CLASS_IDX = 1

def extract_document_images():
    """Load and preprocess the images and save as numpy array file
    """
    worst_doc_dir = os.path.join(FILE_DIR, "worst")
    best_doc_dir = os.path.join(FILE_DIR, "best")

    num_worst = len(os.listdir(worst_doc_dir))
    num_best = len(os.listdir(best_doc_dir))
    num_images = num_worst + num_best
    
    x = np.zeros(shape=(num_images, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), dtype=np.float32)
    y = np.zeros(shape=(num_images), dtype=np.int8)

    cnt = 0
    print("Start reading worst document images!")
    for f in os.listdir(worst_doc_dir):
        img_file = os.path.join(worst_doc_dir, f)
        try:
            img = load_img(img_file, color_mode = 'grayscale', target_size = (IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img).astype('float32')/255
            x[cnt] = img
            y[cnt] = WORST_CLASS_IDX
            cnt += 1
        except:
            print("worst document image %s cannot be read!" % f)

    print("Start reading best document images!")
    for f in os.listdir(best_doc_dir):
        img_file = os.path.join(best_doc_dir, f)
        try:
            img = load_img(img_file, color_mode = 'grayscale', target_size = (IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img).astype('float32')/255
            x[cnt] = img
            y[cnt] = BEST_CLASS_IDX
            cnt += 1
        except:
            print("best document image %s cannot be read!" % f)

    # Dropping not readable image idxs
    x = x[:cnt]
    y = y[:cnt]

    np.save(os.path.join(FILE_DIR, "x.npy"), x)
    np.save(os.path.join(FILE_DIR, "y.npy"), y)

def load_documents(test_size=0.25, extracting_images=False):
    """Loads the data and split into train and test

    Parameters
    ----------
    test_size : float, optional
        percentage for test data, by default 0.33
    extracting_images : bool, optional
        load if files do not exist, by default False

    Returns
    -------
    tuple
        splitted train and test set
    """
    file_x = os.path.join(FILE_DIR, "x.npy")
    file_y = os.path.join(FILE_DIR, "y.npy")

    if not os.path.isfile(file_x) or not os.path.isfile(file_y) or extracting_images:
        extract_document_images()

    x = np.load(file_x)
    y = np.load(file_y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size) 
    return (x_train, y_train), (x_test, y_test)


class Data():
    """Data class to train a neural network
    """
    def __init__(self, test_size=0.30, extracting_images=False):
        """Creates a data instance with the given test size

        Parameters
        ----------
        test_size : float, optional
            percentage for test data, by default 0.33
        extracting_images : bool, optional
            load if files do not exist, by default False
        """
        # Load the dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_documents(
            test_size=test_size, extracting_images=extracting_images)
        self.x_train_ = None
        self.x_val = None
        self.y_train_ = None
        self.y_val = None
        # Convert to float32
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # Save important data attributes as variables
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.train_splitted_size = 0
        self.val_size = 0
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.num_classes = 2 
        self.num_features = self.width * self.height * self.depth
        # Reshape the y data to one hot encoding
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)

    def get_train_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test

    def get_splitted_train_validation_set(self, validation_size=0.33):
        """splits the train set into validation set
            note: instead of this you can use argument:validation_split=x.x in model.fit() method

        Parameters
        ----------
        validation_size : float, optional
            percentage for validation data, by default 0.33

        Returns
        -------
        tuple
            splitted train and validation set
        """
        self.x_train_, self.x_val, self.y_train_, self.y_val =\
            train_test_split(self.x_train, self.y_train, test_size=validation_size)
        self.val_size = self.x_val.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val, self.y_train_, self.y_val

    def data_augmentation(self, augment_size=500):
        """generates additional images and adds them to the training data

        Parameters
        ----------
        augment_size : int, optional
            size of the images that are generated, by default 500
        """
        # Create an instance of the image data generator class
        image_generator = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0.01,
            width_shift_range=0.0,
            height_shift_range=0.0,
            brightness_range=None,
            shear_range=0.0,
            zoom_range=0.02,
            channel_shift_range=0.0,
            # set pixel filling for "free pixels" after rotation nearest: aaaaaaaa | abcd | dddddddd
            fill_mode="nearest", 
            cval=0.0,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0,
            dtype=None)
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
