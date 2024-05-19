import numpy as np
import gzip
import random
from scipy.signal import convolve2d


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

	def SGD_DL(self, w_per_layer, input_image, correct_outputs, LR = 0.001):

		N = len(correct_outputs)													# Numero di campioni.
		h = [[] for _ in range(N)]												# Lista di combinazioni lineari tra pesi e inputs:\
																				# Righe		: 	combinazioni lineari per campione.\
																				# Colonne	: 	combinazioni lineare tra gli input e i pesi\ 
																				# 				leggendo gli strati da sinistra a destra.
																				# Es. 	se ho 5 layer (1 input_L, 3 hidden_L, 1 output_L)\
																				# 		da 25, 20, 20, 20, 5 neuroni, avrò 4 vettori di \
																				# 		combinazioni lineari lunghi rispettivamente\
																				# 		20, 20, 20, 5.

		container_output_prob = [[] for _ in range(N)]							# Lista che conterrà gli output convertiti in valori probabilistici.
		container_weights = [[] for _ in range(N+1)]							# Lista che conterrà tutti gli aggiornamenti dei pesi.
		container_weights[0] = w_per_layer										# Inserisco nella lista dei pesi quelli iniziali forniti dall'utente.

		# ciclo ad ogni campione ---------------------------------------------- #
		for k in range(N):														
			
			a = [input_image[:, :, k].flatten()]								# Salvo i valori iniziali dei neuroni assegnati in input\
																				# relativi al campione k-esimo.	
																						
			# ciclo di training FORWARD --------------------------------------- #
			for i, w in enumerate(container_weights[k]):						# Ciclo in cui vengono applicati i pesi strato per strato.
				h[k] += [w @ a[i]]												# Combinazione lineare tra pesi e inputs con dimensione\ 
																				# finale pari al numero di neuroni nel layer successivo.
				if i != (len(container_weights[k]) - 1):						# Applico la funzione di attivazione ReLU solamente ai layer interni.
					a += [self.ReLU(h[k][i])]									# Salvo gli output (ReLU(h)) come valori di input (a) per il layer\ 
																				# successivo.
				else:
					a += [self.Softmax(h[k][i])]								# All'ultimo output applico la funzione Softmax per trasformare il\
																				# vettore reale in vettore di probabilità la cui somma totale è pari a 1.
					container_output_prob[k] = a[-1]							# Salvo i vettori probabilità nella lista precedentemente definita.
			
			# ciclo di training BACKWARD -------------------------------------- #
			for i in range(len(container_weights[k])):									
				if i == 0:														# Al primo ciclo definisco la funzione di costo derivata utilizzando\ 
																				# come metrica la MSE.
					E = correct_outputs[k,:]									# Vettore di output attesi sull'output layer.
					P = h[k][-1]												# Vettore di output predetti: Non considero l'applicazione della\ 
																				# Softmax, poiché è una funzione utile solamente per convertire i\
																				# risultati in probabilità e non contribuisce al processo di correzione\
																				# dei pesi.
					f_cost_der = 2 * (P - E)									# Vettore funzione di costo derivata.
					delta = f_cost_der											# Prima delta per la retropropagazione degli errori.
				else:
					f_cost_der = container_weights[k][-i].T @ delta				# Applico a ritroso i pesi alle funzioni di costo sui layer precedenti.
																				# Es. Prodotto interno tra i pesi sull'ultimo Hidden Layer\ 
																				# (di dimensioni MxN) e la delta (di dimensioni (Mx1)): dim finale = Nx1.						
					ReLU_der = np.where(a[-(i+1)] > 0, 1, 0)					# Derivata del vettore ReLU. Es. di prima: ReLU dell'ultimo HL (dim = Nx1).
					delta = ReLU_der * f_cost_der								# Delta relativa al layer in esame (dim = Nx1).
				dW = LR * np.outer(delta, a[-(i+2)])							# Calcolo la correzione dei pesi, che corrisponde al prodotto esterno tra\
																				# delta (Nx1) e gli input (Sx1): dim finale = NxS.
				w = np.copy(container_weights[k][-(i+1)])						# Copio il valore degli ultimi pesi in esame per evitare di sovrascrivere\
																				# i pesi contenuti nella lista durante la correzione.
				w -= dW															# Aggiorno i pesi.
				container_weights[k+1].insert(0, w)								# Salvo i pesi nella lista definita in precedenza. L'ultima riga contiene\
																				# i pesi finali.

		return (container_weights, container_output_prob)
	
# ----------------------------------------------------------------------------- #
	
	def Test_DL(self, w_per_layer, input_image, correct_outputs):
		
		N = len(correct_outputs)
		container_output_prob = [[] for _ in range(N)]							

		# ciclo ad ogni campione 
		for k in range(N):														
			a = input_image[:, :, k].flatten()					
																																					
			for i, w in enumerate(w_per_layer):								
				h = w @ a											 
																
				if i != (len(w_per_layer) - 1):							
					a = self.ReLU(h)								
																
				else:
					a = self.Softmax(h)																		
					container_output_prob[k] = a								
		
		return container_output_prob
	
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
		numFilters, wrow, wcol = W.shape
		_, xrow, xcol = x.shape

		yrow = xrow - wrow + 1
		ycol = xcol - wcol + 1

		y = np.zeros((numFilters, yrow, ycol))

		for k in range(numFilters):
			filter = W[k, :, :]
			filter = np.rot90(filter, 2)
			y[k, :, :] = convolve2d(x[k, :, :], filter, mode='valid')

		return y

# ----------------------------------------------------------------------------- #
