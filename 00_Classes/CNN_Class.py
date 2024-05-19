import numpy as np
import gzip
import random
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


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

	def load_MNIST_labels(self, filename):
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

	def Rng(self, x):
		np.random.seed(x)
		random.seed(x)

# ----------------------------------------------------------------------------- #

	def SGD_MnistConv(self, W1, W5, Wo, X, D, LRA = 0.001, LRB = 0.95):
		momentum_1 = np.zeros_like(W1)								# Definisco i contenitori per le correzioni dei pesi.
		momentum_5 = np.zeros_like(W5)
		momentum_o = np.zeros_like(Wo)


		N = len(D)													
		bsize = 100  												# Definisco i batch (set di immagini).			
		blist = range(0, N - bsize + 1, bsize)						

		for batch in blist:
			dW1 = np.zeros_like(W1)									# Definisco i contenitori per le correzioni dei pesi moltiplicati per LRA.
			dW5 = np.zeros_like(W5)
			dWo = np.zeros_like(Wo)


			begin = batch
			for k in range(begin, begin + bsize):					# Valuto solo le immagini nel range in esame.
				
				# Forward
				x = X[k, :, :]										# Estraggo la singola immagine (28x28).
				y1 = self.Conv(x, W1)								# Applico il filtro di convoluzione. dim(y1) = 20x20x20
				y2 = self.ReLU(y1)									# Applico la ReLU per schiacciare i dati pesati. dim(y2) = 20x20x20
				y3 = self.Pool(y2)									# Applico il filtro di Pooling. dim(y3) = 20x10x10
				y4 = y3.flatten()									# Estendo la matrice in un unico vettore. dim(y4) = 2000
				y5 = self.ReLU(W5 @ y4)								# Applico la ReLU per schiacciare i dati pesati. dim(y5) = 100
				v = Wo @ y5											# Calcolo la combinazione lineare per ottenere i dati di output. dim(y1) = 10
				predicted = self.Softmax(v)							# All'ultimo output applico la funzione Softmax per trasformare il\
																	# vettore reale in vettore di probabilità la cui somma totale è pari a 1.

				# Backward
				expected = np.zeros(10)								# Creo il vettore di valori attesi, lungo quanto i neuroni dell'output layer.
				expected[D[k]] = 1									# Metto ad 1 la la cella delle 10 corrispondente alla cifra aspettata.
		
				f_cost_der = 2 * (predicted - expected)				# Derivata della funzione di costo.
				delta = f_cost_der		
				dWo += np.outer(delta, y5)							

				f_cost_der_5 = Wo.T @ delta																						
				ReLU_der_5 = np.where(y5 > 0, 1, 0)				
				delta_5 = ReLU_der_5 * f_cost_der_5	
				dW5 += np.outer(delta_5, y4)

				f_cost_der_4 = W5.T @ delta_5
				f_cost_der_3 = f_cost_der_4.reshape(y3.shape)
				f_cost_der_2 = np.zeros_like(y2)
				
				W3 = np.ones_like(y2) / (2 * 2)

				for c in range(20):
					f_cost_der_2[c, :, :] = np.kron(f_cost_der_3[c, :, :], np.ones((2, 2))) * W3[c, :, :]

				ReLU_der_2 = np.where(y2 > 0, 1, 0)	
				delta_2 = ReLU_der_2 * f_cost_der_2

				delta_1 = np.zeros_like(W1)
				for c in range(20):
					delta_1[c, :, :] = np.rot90(convolve2d(x, np.rot90(delta_2[c, :, :], 2), mode='valid'), 2)
				dW1 += delta_1

			dW1 /= bsize
			dW5 /= bsize
			dWo /= bsize

			momentum_1 = LRA * dW1 + LRB * momentum_1
			W1 -= momentum_1

			momentum_5 = LRA * dW5 + LRB * momentum_5
			W5 -= momentum_5

			momentum_o = LRA * dWo + LRB * momentum_o
			Wo -= momentum_o

		return W1, W5, Wo															

	
# ----------------------------------------------------------------------------- #

	def Pool(self, x):
		numFilters, xrow, xcol = x.shape
		y = np.zeros((numFilters, xrow // 2, xcol // 2))  # Divisione intera per 2
		
		for k in range(numFilters):
			filter = np.ones((2, 2)) / (2 * 2)            # pooling media
			image = convolve2d(x[k, :, :], filter, mode='valid')
			y[k, :, :] = image[0::2, 0::2]  # prendo la media e la metto nel pixel che scelgo ogni 4
			
		return y

# ----------------------------------------------------------------------------- #

	def Conv(self, x, W):
		num_filters, wrow, wcol = W.shape
		xrow, xcol = x.shape

		yrow = xrow - wrow + 1
		ycol = xcol - wcol + 1

		y = np.zeros((num_filters, yrow, ycol))

		for k in range(num_filters):
			filter = W[k, :, :]
			filter = np.rot90(filter, 2)
			y[k, :, :] = convolve2d(x, filter, mode='valid')

		return y

# ----------------------------------------------------------------------------- #

	def display_network(self, A, title, opt_normalize=True, opt_graycolor=True, cols=None, opt_colmajor=False):
		# Disabilita i warning (equivalente in Python)
		import warnings
		warnings.filterwarnings("ignore")

		# Sottrae la media
		A = A - np.mean(A)

		# Imposta la mappa di colori in scala di grigi se richiesto
		if opt_graycolor:
			plt.set_cmap('gray')

		L, M = A.shape
		sz = int(np.sqrt(L))
		buf = 1

		if cols is None:
			if int(np.sqrt(M))**2 != M:
				n = int(np.ceil(np.sqrt(M)))
				while M % n != 0 and n < 1.2 * np.sqrt(M):
					n += 1
				m = int(np.ceil(M / n))
			else:
				n = int(np.sqrt(M))
				m = n
		else:
			n = cols
			m = int(np.ceil(M / n))

		array = -np.ones((buf + m * (sz + buf), buf + n * (sz + buf)))

		if not opt_colmajor:
			k = 0
			for i in range(m):
				for j in range(n):
					if k >= M:
						continue
					clim = np.max(np.abs(A[:, k]))
					if opt_normalize:
						array[buf + i * (sz + buf):buf + i * (sz + buf) + sz,
							buf + j * (sz + buf):buf + j * (sz + buf) + sz] = A[:, k].reshape(sz, sz) / clim
					else:
						array[buf + i * (sz + buf):buf + i * (sz + buf) + sz,
							buf + j * (sz + buf):buf + j * (sz + buf) + sz] = A[:, k].reshape(sz, sz) / np.max(np.abs(A))
					k += 1
		else:
			k = 0
			for j in range(n):
				for i in range(m):
					if k >= M:
						continue
					clim = np.max(np.abs(A[:, k]))
					if opt_normalize:
						array[buf + i * (sz + buf):buf + i * (sz + buf) + sz,
							buf + j * (sz + buf):buf + j * (sz + buf) + sz] = A[:, k].reshape(sz, sz) / clim
					else:
						array[buf + i * (sz + buf):buf + i * (sz + buf) + sz,
							buf + j * (sz + buf):buf + j * (sz + buf) + sz] = A[:, k].reshape(sz, sz)
					k += 1

		plt.imshow(array, vmin=-1, vmax=1)
		plt.axis('off')
		plt.title(title)
		plt.show()

		# Riabilita i warning
		warnings.filterwarnings("default")

# ----------------------------------------------------------------------------- #
