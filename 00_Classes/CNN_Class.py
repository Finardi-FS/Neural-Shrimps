import numpy as np
import gzip
import random

class MyCNN:

	def __init__(self):
		pass
	
	# ----------------------------------------------------------------------------- #

	def load_MNIST_images(self, filename):
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

# ----------------------------------------------------------------------------- #

	def loadMNISTLabels(self, filename):
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

	def ReLU(self, x):
		return np.maximum(0, x)
	
# ----------------------------------------------------------------------------- #

	def Softmax(self, x: np.array):
		max_x = np.max(x)
		
		# Verifica se max_x supera una soglia per evitare l'overflow
		if max_x > 709:  # 709 è scelto perché np.exp(710) risulta in overflow
			x = x - max_x
		
		exp = np.exp(x)
		return exp / np.sum(exp)

# ----------------------------------------------------------------------------- #

	def rng(self, x):
		np.random.seed(x)
		random.seed(x)