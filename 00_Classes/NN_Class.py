import numpy as np
import pandas as pd
from sklearn.model_selection import KFold # divide il set di dati in k fold (sottogruppi) di dimensioni uguali.
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import jupyprint.jupyprint as jp


# ///////////////////////////////////////////////////////////////////////////// #
# Class for Neural Nerwork
# ///////////////////////////////////////////////////////////////////////////// #

class MyNN:

	def __init__(self):
		self._learning_rate = 0.9


# ----------------------------------------------------------------------------- #


	def Sigmoid(self, x):
		return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------- #


	def SGD_method(self, weights, dataset, correct_outputs, LRA = None, LRB = None, LRD = False, momentum = None):
																	# weight:           vettore dei pesi relativi alle feature
																	# f_input:          dataset contenente i valori delle feature per ciascun campione
																	# correct_output:   vettore dei valori attesi per ciascun campione
																	# LRA: learning rate principale	 
																	# LRB: learning rate secondario
																	# LRD: opzione learning rate dinamico
		
		if not LRA:
			LR = self._learning_rate

		for k in range(len(dataset)):                                   # ciclo sui campioni
			
			sample = dataset[k]                                         # prendo i valori delle feature del k-esimo campione
			expected = correct_outputs[k]                               # prendo il valore atteso del k-esimo campione
			lin_comb = weights @ sample                                 # combinazione lineare delle feature pesate per il k-esimo campione
			predicted = self.Sigmoid(lin_comb)                          # normalizzazione della comb. lin. attraverso la funzione Sigmoid         
			
			cost_function_der = 2 * (predicted - expected)              # derivata della funzione di costo (rispetto alla variabile predicted) 
																		# relativa alla metrica Squared Error Cost                         
			sigmoid_der = predicted * (1 - predicted)                   # derivata del sigmoid rispetto a lin_comb
			lc_der = sample                                             # vettore di derivate di lin_comb rispetto ai pesi w (e al bias b)
			
			if LRD:
				momentum = LRA * cost_function_der * sigmoid_der * lc_der + LRB * momentum
				dWeights = momentum
			else:
				dWeights = LRA * cost_function_der * sigmoid_der * lc_der    # termine differenziale di aggiustamento per ricavare i nuovi pesi
			
			weights -= dWeights                                         # correzione dei pesi
		
		self._final_weights = weights
		
		return (weights, momentum)


# ----------------------------------------------------------------------------- #


	def training(self, weights, inputs, correct_outputs, epochs, LRA = None, LRB = None, LRD = False): 	# LRA : learning rate principale	 
																										# LRB : learning rate secondario
																										# LRD : opzione learning rate dinamico

		self._weights_per_epoch = np.array([weights])
		weights = np.copy(weights)

		if LRD:
			momentum = 0

		for e in range(epochs): #epochs-1
			if LRD:
				weights, momentum = self.SGD_method(weights, inputs, correct_outputs, LRA, LRB, LRD, momentum)
			else:
				weights = self.SGD_method(weights, inputs, correct_outputs, LRA)[0]
			
			self._weights_per_epoch = np.vstack((self._weights_per_epoch, weights))
		lin_comb = (self._weights_per_epoch @ inputs.T).T
		P = self.Sigmoid(lin_comb)
		E = np.tile(correct_outputs, (epochs+1,1)).T

		return {
				"container_weights" : self._weights_per_epoch,
				"correct_outputs"  	: correct_outputs,
				"epochs"            : epochs,
				"inputs"            : inputs,
				"P"					: P,
				"E"					: E
			}


# ----------------------------------------------------------------------------- #


	def training_batch(self, weights, inputs, correct_outputs, epochs, LRA = None, LRB = None, LRD = False): 	# LRA : learning rate principale	 
																										# LRB : learning rate secondario
																										# LRD : opzione learning rate dinamico

		self._weights_per_epoch = np.array([weights])
		weights = np.copy(weights)

		if LRD:
			momentum = 0

		for e in range(epochs): #epochs-1
			if LRD:
				weights, momentum = self.Batch_method(weights, inputs, correct_outputs, LRA, LRB, LRD, momentum)
			else:
				weights = self.Batch_method(weights, inputs, correct_outputs, LRA)[0]
			
			self._weights_per_epoch = np.vstack((self._weights_per_epoch, weights))
		lin_comb = (self._weights_per_epoch @ inputs.T).T
		P = self.Sigmoid(lin_comb)
		E = np.tile(correct_outputs, (epochs+1,1)).T

		return {
				"container_weights" : self._weights_per_epoch,
				"correct_outputs"  	: correct_outputs,
				"epochs"            : epochs,
				"inputs"            : inputs,
				"P"					: P,
				"E"					: E
			}


# ----------------------------------------------------------------------------- #


	def Batch_method(self, weights, dataset, correct_outputs, LRA = None, LRB = None, LRD = False, momentum = None):
																	# weight:           vettore dei pesi relativi alle feature
																	# f_input:          dataset contenente i valori delle feature per ciascun campione
																	# correct_output:   vettore dei valori attesi per ciascun campione
																	# LRA: learning rate principale	 
																	# LRB: learning rate secondario
																	# LRD: opzione learning rate dinamico
		dWeights = 0
		
		if not LRA:
			LR = self._learning_rate

		for k in range(len(dataset)):                                   # ciclo sui campioni
			
			sample = dataset[k]                                         # prendo i valori delle feature del k-esimo campione
			expected = correct_outputs[k]                               # prendo il valore atteso del k-esimo campione
			lin_comb = weights @ sample                                 # combinazione lineare delle feature pesate per il k-esimo campione
			predicted = self.Sigmoid(lin_comb)                          # normalizzazione della comb. lin. attraverso la funzione Sigmoid         
			
			cost_function_der = 2 * (predicted - expected)              # derivata della funzione di costo (rispetto alla variabile predicted) 
																		# relativa alla metrica Squared Error Cost                         
			sigmoid_der = predicted * (1 - predicted)                   # derivata del sigmoid rispetto a lin_comb
			lc_der = sample                                             # vettore di derivate di lin_comb rispetto ai pesi w (e al bias b)
			
			if LRD:
				momentum = LRA * cost_function_der * sigmoid_der * lc_der + LRB * momentum
				dWeights += momentum
			else:
				dWeights += LRA * cost_function_der * sigmoid_der * lc_der    # termine differenziale di aggiustamento per ricavare i nuovi pesi
	
		
		weights -= dWeights / len(dataset)                                         # correzione dei pesi
		
		self._final_weights = weights
		
		return (weights, momentum)
	

# ----------------------------------------------------------------------------- #


	# def plt_epochs(self, results, idx_sample):    

	# 	def cost_func(x, x0):
	# 		return (x - x0)**2

	# 	plt.figure(figsize=(15,8))

	# 	P = self.Sigmoid(results['container_weights'] @ results['inputs'][idx_sample])
	# 	E = np.ones(len(P)) * results['correct_outputs'][idx_sample]

	# 	plt.subplot(1,2,1)
	# 	x = np.linspace(0, P[0]*1.1, 1000)  
	# 	x0 = results['correct_outputs'][idx_sample]
	# 	plt.grid()
	# 	plt.plot(x, cost_func(x, x0))
	# 	plt.scatter(E[0],0, color='red', linewidth=5)
	# 	plt.plot(P, cost_func(P, E), marker = 'o')
	# 	plt.title(f'Cost function minimization (sample #{idx_sample})')

	# 	plt.subplot(1,2,2)
	# 	plt.grid()
	# 	plt.plot(np.arange(1, results['epochs']+2), 2*(P - E))
	# 	plt.title(f'(sample #{idx_sample})')
	# 	plt.show()


	def plt_epochs(self, results, idx_sample):    

		def cost_func(x, x0):
			return (x - x0)**2

		plt.figure(figsize=(16, 8))

		P = self.Sigmoid(results['container_weights'] @ results['inputs'][idx_sample])
		E = np.ones(len(P)) * results['correct_outputs'][idx_sample]

		# Plot della funzione di costo
		plt.subplot(1, 2, 1)
		x = np.linspace(0, P[0]*1.1, 1000)
		x0 = results['correct_outputs'][idx_sample]
		plt.grid(True, linestyle='--', linewidth=0.5)
		plt.plot(x, cost_func(x, x0), label='Funzione di costo', linewidth = 2)
		plt.scatter(E[0], 0, s=100, color = 'red', zorder=5, label='Output corretto')
		plt.plot(P, cost_func(P, E), marker='o', label='Output stimato', linewidth = 2, ms=8)
		plt.title('Funzione di costo', fontsize=22)
		plt.xlabel('Output stimato', fontsize=18)
		plt.ylabel('Costo', fontsize=18)
		plt.legend(fontsize=18)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		
		# Plot della loss function in funzione delle epoche
		plt.subplot(1, 2, 2)
		plt.grid(True, linestyle='--', linewidth=0.5)
		plt.plot(np.arange(1, results['epochs'] + 2), 2 * (P - E), label='Loss Function', color='purple', marker='o', linewidth = 2, ms =8)
		plt.title(f'Loss Function al variare delle epoche', fontsize=22)
		plt.xlabel('Epoche', fontsize=18)
		plt.ylabel('Loss', fontsize=18)
		plt.legend(fontsize=18)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)

		plt.tight_layout()
		plt.show()


# ----------------------------------------------------------------------------- #


	def data_folding(self, K, Nsamples, show_kfold = False):
		
		if K == 1:
			indices = [np.ones(Nsamples, dtype=int)]
			
		else:
			folds = [fold for others, fold in KFold(n_splits = K, shuffle = True).split(np.arange(Nsamples))]

			def indcs(folds, N):
				indices = np.arange(N)
				for i in range(K):
					for j in folds[i]:
						indices[j] = i
				return indices

			indices = indcs(folds, Nsamples)

		if show_kfold:
			jp('### Indici delle fold associate a ciascun campione:')
			jp(indices)
		return (K, indices)


# ----------------------------------------------------------------------------- #


	def table_kfolds(self, dic: dict):
		jp(pd.DataFrame(
			dic.values(), 
			index = dic.keys(), 
			columns = ['(K: {})'.format(str(i)+' of '+str(len(dic['FP']))) 
			  			for i in range(1,len(dic['FP'])+1)]))


# ----------------------------------------------------------------------------- #


	def testing(self, inputs, correct_outputs, weights = None, K = 1):
		if weights is None:
			weights = self._weights_per_epoch[-1]
		pred_outputs 	= np.empty((1,0))
		metrics			= {
			'accuracy'    : 0,
			'sensitivity' : 0,
			'specificity' : 0,
			'FP'          : 0,
			'FN'          : 0
		}			


		for k in range(len(inputs)):
			weighted_sum = weights @ inputs[k]  # Prodotto scalare tra pesi allenati e input
			predicted = self.Sigmoid(weighted_sum)
			pred_outputs = np.append(pred_outputs, predicted) 

			# Caso 0 -> Negative
			if correct_outputs[k] == 0 and predicted < 0.5:
				metrics['accuracy'] += 1 / len(correct_outputs)
				metrics['specificity'] += 1 / (len(correct_outputs)-np.sum(correct_outputs))

			# Caso 1 -> Positive
			if correct_outputs[k] == 1 and predicted >= 0.5:
				metrics['accuracy'] += 1 / len(correct_outputs)
				metrics['sensitivity'] += 1 / np.sum(correct_outputs)

			# Caso 1 -> Negative
			if correct_outputs[k] == 1 and predicted < 0.5:
				metrics['FN'] += 1 / np.sum(correct_outputs)

			# Caso 0 -> Positive
			if correct_outputs[k] == 0 and predicted >= 0.5:
				metrics['FP'] += 1 / (len(correct_outputs)-np.sum(correct_outputs))

		return (inputs, correct_outputs, pred_outputs, metrics)


# ----------------------------------------------------------------------------- #


	def trend_over_epochs(self, inputs, correct_outputs, weights_per_epoch):
		P = np.empty((len(inputs),0))
		for i in range(1, len(weights_per_epoch)+1):
			pred_per_sample = self.testing(inputs, correct_outputs, weights_per_epoch[:i][-1])[2]
			P = np.hstack((P, np.array([pred_per_sample]).T))
		E = np.tile(correct_outputs, (len(weights_per_epoch),1)).T
		return {
			"P" : P,
			"E" : E 
		}


# ----------------------------------------------------------------------------- #


	def avg_std_kfolds(self, dic: dict, table = False):
		acc_avg = np.sum(dic['Accuracy'])/len(dic['Accuracy'])
		acc_std = np.std(dic['Accuracy'])

		sen_avg = np.sum(dic['Sensitivity'])/len(dic['Sensitivity'])
		sen_std = np.std(dic['Sensitivity'])

		spe_avg = np.sum(dic['Specificity'])/len(dic['Specificity'])
		spe_std = np.std(dic['Specificity'])

		FN_avg = np.sum(dic['FN'])/len(dic['FN'])
		FN_std = np.std(dic['FN'])

		FP_avg = np.sum(dic['FP'])/len(dic['FP'])
		FP_std = np.std(dic['FP'])

		avg_results = {
			'Avg': {'Accuracy':         acc_avg,
					'Sensitivity':      sen_avg,
					'Specificity':      spe_avg,
					'FN':   			FN_avg,
					'FP':   			FP_avg
					},
			'Std': {'Accuracy':         acc_std,
					'Sensitivity':      sen_std,
					'Specificity':      spe_std,
					'FN':   			FN_std,
					'FP':   			FP_std
		   			}
			}
		if table:
			jp(pd.DataFrame({
				'Avg': avg_results['Avg'].values(),
				'Std': avg_results['Std'].values()
				}, 
				index = avg_results['Avg'].keys()))
		return avg_results
		

# ----------------------------------------------------------------------------- #


	def ROC_curve(self, labels, prob_pred):
		# Calcola la curva ROC per ciascuna fold e calcola l'AUC
		mean_fpr = np.linspace(0, 1, 100)
		tprs = []
		aucs = []
		fpr, tpr, _ = roc_curve(labels, prob_pred)
		tprs.append(np.interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)

		# Calcola la media e la deviazione standard delle curve ROC e degli AUC
		mean_tpr = np.mean(tprs, axis=0)
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)

		# Disegna la curva ROC media
		plt.figure(figsize=(10, 8))
		plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
		plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', linewidth=2)

		# Imposta le etichette e la legenda
		plt.xlabel('False Positive Rate', fontsize=18)
		plt.ylabel('True Positive Rate', fontsize=18)
		plt.title('Curva ROC', fontsize=22)
		plt.legend(loc='lower right', fontsize=18)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.grid(True, linestyle='--', linewidth=0.5)
		plt.tight_layout()
		plt.show()

		return {'fpr': mean_fpr, 'tpr': mean_tpr}