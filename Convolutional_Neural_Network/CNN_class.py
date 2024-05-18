import numpy as np
import gzip

def load_MNIST_images(filename):
    with gzip.open(filename, 'r') as f:
        # first 4 bytes is a magic number
        magic_number    = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count     = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count       = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count    = int.from_bytes(f.read(4), 'big')
        
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        images = np.frombuffer(f.read(), dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
    return images


def loadMNISTLabels(filename):
    with gzip.open(filename, 'r') as f:
        # first 4 bytes is a magic number
        magic_number    = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count     = int.from_bytes(f.read(4), 'big')
        
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data      = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
    return labels

# ----------------------------------------------------------------------------- #

def ReLU(x):
    return np.maximum(0, x)
	
# ----------------------------------------------------------------------------- #

def Softmax(x: np.array):
    # exp = np.exp(x - np.max(x))  # Sottrarre il massimo valore per stabilizzare l'output
    # return exp / np.sum(exp)
    exp = np.exp(x)
    return exp / np.sum(exp)

# ----------------------------------------------------------------------------- #

def rng(x):
    np.random.seed(x)
    random.seed(x)