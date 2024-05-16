import numpy as np
import pandas as pd
from sklearn.model_selection import KFold # divide il set di dati in k fold (sottogruppi) di dimensioni uguali.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import jupyprint.jupyprint as jp







# ///////////////////////////////////////////////////////////////////////////// #
# Class for Principal Component Analysis (PCA)
# ///////////////////////////////////////////////////////////////////////////// #

class PCA_Class:

	def __init__(self, data_int, data_ext = None, feature_names = None):

		self._dict = {}
		self._dict['cov_mat']               = np.corrcoef(data_int, rowvar=False)
		evl, evc                            = np.linalg.eig(self._dict['cov_mat'])
		self._dict['sorted_idcs']           = np.argsort(evl)[::-1]
		self._dict['eval']                  = evl[self._dict['sorted_idcs']]
		self._dict['evec']                  = evc[:,self._dict['sorted_idcs']]
		self._dict['data_int']              = data_int[:,self._dict['sorted_idcs']]
		if data_ext is not None:
			self._dict['data_ext']          = data_ext[:,self._dict['sorted_idcs']]
		if feature_names is not None:
			self._dict['sorted_feat_names'] = list(np.array(feature_names)[self._dict['sorted_idcs']])
		else:
			self._dict['sorted_feat_names'] = list(np.array([f'f_'+str(i) for i in np.arange(len(data_int))])[self._dict['sorted_idcs']])
		self._dict['sum_eval']              = np.round(np.sum(self._dict['eval']), 1)
		self._dict['eval_var']              = np.abs(self._dict['eval']/self._dict['sum_eval'])
		self._dict['cumulative_var']        = np.cumsum(self._dict['eval_var'])

	def get_results(self):
		return self.dict

	def get_cov_mat(self, show_tb = False):
		if show_tb:
			pd.options.display.float_format = '{:,.3e}'.format
			jp(pd.DataFrame(self._dict['cov_mat']))
		return self._dict['cov_mat']
		
	def get_eval(self, show_tb = False):
		if show_tb:
			pd.options.display.float_format = '{:,.3e}'.format
			jp(pd.DataFrame(self._dict['eval']))
		return self._dict['eval']
	
	def get_evec(self, show_tb = False):
		if show_tb:
			pd.options.display.float_format = '{:,.3e}'.format
			jp(pd.DataFrame(self._dict['evec']))
		return self._dict['evec']
	
	def show_eval_percent(self):
		cumulative = self._dict['cumulative_var']
		eval_var = self._dict['eval_var']
		eval = self._dict['eval']

		Tab = np.column_stack((	['{:0.3e}'.format(num) for num in eval], 
								['{}%'.format(np.round(num*100,3)) for num in eval_var], 
								['{}%'.format(np.round(num*100,3)) for num in cumulative]))
		end_line = np.vstack([	['-','-','-'], 
								[self._dict['sum_eval'], '{}%'.format(np.sum(eval_var)*100 // 1), '-']])
		df = pd.DataFrame(	np.vstack([Tab, end_line]), 
							index = [f'λ{str(i)}' for i in range(len(eval))]+['-','TOT'], 
							columns = ['Eigenvalue', 'Percentage', 'Cumulative'])
		jp(df)

		plt.figure()
		plt.title('Percentuale degli autovalori sul totale')
		plt.xticks(range(1,len(eval)+1), ['λ'+str(i) for i in range(len(eval))])
		plt.bar(range(1,len(eval)+1),eval_var, label = '% Variance')
		plt.plot(range(1,len(eval)+1),cumulative, linewidth=1, marker='o', color = 'orange', label = 'Cumulative Variance')
		plt.legend()
		plt.grid()
		plt.show()

	def get_PCA_data(self, t, show_tb = False, raw_names_int = None, raw_names_ext = None):
		"""t: cumulative variance threshold"""
		cumulative = self._dict['cumulative_var']
		
		self._PCA_data = {}

		# prendo gli autovalori che spieghino un tot percentuale (t) della varianza
		self._PCA_data['pca_eval'] = self._dict['eval'][cumulative < t] 
		# prendo i corrispettivi autovettori
		self._PCA_data['pca_evec'] = self._dict['evec'][:,cumulative < t] 

		# Prodotto scalare tra matrice di dati (righe features, colonne stati) 
  		# e matrice di autovettori principali disposti in colonna
		self._PCA_data['pca_data_int'] = self._dict['data_int'] @ self._PCA_data['pca_evec'] 
		if 'data_ext' in self._dict.keys():
			self._PCA_data['pca_data_ext'] = self._dict['data_ext'] @ self._PCA_data['pca_evec'] 

		



		if show_tb:
			if raw_names_int is None:
				raw_names_int = np.arange(len(self._PCA_data['pca_data_int']))
			if (raw_names_ext is None) and ('data_ext' in self._dict.keys()):
				raw_names_ext = np.arange(len(self._PCA_data['pca_data_ext']))

			jp('### PCA - Eigenvalues')
			jp(pd.DataFrame(self._PCA_data['pca_eval'].reshape(1,-1), index = ['pc λ'], 
				   			columns = ['λ'+str(i) for i in range(len(self._PCA_data['pca_eval']))]))
			jp('### PCA - Internal data')
			jp(pd.DataFrame(self._PCA_data['pca_data_int'], index = raw_names_int, 
				   			columns = ['Y'+str(i) for i in range(len(self._PCA_data['pca_eval']))]))
			if 'data_ext' in self._dict.keys():
				jp('### PCA - External data')
				jp(pd.DataFrame(self._PCA_data['pca_data_ext'], index = raw_names_ext, 
				   				columns = ['Y'+str(i) for i in range(len(self._PCA_data['pca_eval']))]))				
		return self._PCA_data

	def get_PCA_data_ext(self, data_ext = None, show_tb = False, raw_names_ext = None):

		try:
			if data_ext is not None:	
				self._dict['data_ext'] = data_ext[:,self._dict['sorted_idcs']]
				self._PCA_data['pca_data_ext'] = self._dict['data_ext'] @ self._PCA_data['pca_evec']

			if show_tb:
				if raw_names_ext is None:
					raw_names_ext = np.arange(len(self._PCA_data['pca_data_ext']))
				jp('### PCA - External data')	
				jp(pd.DataFrame(self._PCA_data['pca_data_ext'], index = raw_names_ext, 
								columns = ['Y'+str(i) for i in range(len(self._PCA_data['pca_eval']))]))	
			
			return self._PCA_data['pca_data_ext']
		except:
			print('Eseguire prima la PCA sul dataset interno.')

	def show_evec(self, idx_evec: int):
		try:
			evec = self._dict['evec']			
			plt.figure()
			plt.subplots_adjust(bottom=0.5)
			plt.title(f'Coefficienti autovettore {idx_evec} in valore assoluto')
			plt.xticks(range(0,len(evec)), self._dict['sorted_feat_names'], rotation=45, ha='right')
			plt.plot(np.abs(evec[idx_evec]))
			plt.grid()
			plt.show()
		except:
			print(f"L'autovettore con indice {idx_evec} non esiste.")




# ///////////////////////////////////////////////////////////////////////////// #
# Class for Machine Learning 
# ///////////////////////////////////////////////////////////////////////////// #

class ML_Class:

	def __init__(self, data, labels):
		self._data = data
		self._labels = labels
		self._numVar = self._data.shape[1]
		self._N = len(self._labels)
		self._original_indices = np.arange(self._N)
		

# ----------------------------------------------------------------------------- #


	def get_data(self):
		return {'data': self._data, 'labels': self._labels}    


# ----------------------------------------------------------------------------- #


	def data_folding(self, K, show_kfold = False):
		
		folds = [fold for others, fold in KFold(n_splits = K, shuffle = True).split(self._original_indices)]

		def indxs(folds, N):
			indices = np.arange(N)
			for i in range(K):
				for j in folds[i]:
					indices[j] = i
			return indices

		indices = indxs(folds, self._N)

		if show_kfold:
			jp('### Indici delle fold associate a ciascun campione:')
			jp(indices)
		return (K, indices)


# ----------------------------------------------------------------------------- #


	def train_test(self, trained_model, folding: data_folding, numVar = None, kappa = True):
		if numVar:
			self._numVar = numVar
		
		# Metrics
		K, index = folding
		
		accuracy = np.zeros(K)
		sensitivity = np.zeros(K)
		specificity = np.zeros(K)
		FN = np.zeros(K)
		FP = np.zeros(K)
		
		models = {}
		
		self._prob_pred = np.zeros(self._N) # Per salvare probabilità classificazione
		
		if kappa:
			print('k =', K)
		
		for ind in range(K):
			temp_train_set = np.ones(self._data.shape[0], dtype=bool)
			temp_train_set[index == ind] = False
			temp_test_set = ~temp_train_set
			
			temp_train_indices = self._original_indices[temp_train_set]
			temp_test_indices = self._original_indices[temp_test_set]
			
			temp_train_labels = self._labels[temp_train_indices]
			temp_test_labels = self._labels[temp_test_indices]
			
			temp_train_data = self._data[temp_train_indices, :]
			temp_test_data = self._data[temp_test_indices, :]
			
			try:
				pc_tr_data_non_scaled = np.squeeze(temp_train_data[:, :self._numVar])
				
				scaler = StandardScaler()
				pc_tr_data = scaler.fit_transform(pc_tr_data_non_scaled)
				
				models[str(ind)] = trained_model.fit(pc_tr_data, temp_train_labels)
			
			except Exception as e:
				print('Error optimization preprocessing')
				msgString = str(e)
			
			# TESTING phase
			temp_test_data_non_scaled = np.squeeze(temp_test_data[:, :self._numVar])
			temp_te_data = scaler.fit_transform(temp_test_data_non_scaled)
			
			prob_pred_fold = np.zeros(temp_te_data.shape[0])
			
			for subject in range(temp_te_data.shape[0]):
			
				pc_temp_tdata = temp_te_data[subject, :]
				
				temp_tlabel = temp_test_labels[subject]
				
				if pc_temp_tdata.shape[0] == 0:
						print('No data!')
				
				else:
					pc_te_data = pc_temp_tdata[:self._numVar]
					
					# Predizione delle probabilità invece delle etichette di classe
					prob_pred_fold[subject] = trained_model.predict_proba([pc_te_data])[0][1]

					# Label predetta dal modello
					class_pred = trained_model.predict([pc_te_data])
					
					#print(class_pred)
					class_pred = class_pred.astype(float)
					
					# Caso 0 -> Negative
					if temp_tlabel == 0 and class_pred == 0:
						accuracy[ind] += 1 / temp_test_labels.shape[0]
						specificity[ind] += 1 / (temp_test_labels.shape[0]-np.sum(temp_test_labels))
					
					# Caso 1 -> Positive
					if temp_tlabel == 1 and class_pred == 1:
						accuracy[ind] += 1 / temp_test_labels.shape[0]
						sensitivity[ind] += 1 / np.sum(temp_test_labels)
					
					# Caso 1 -> Negative
					if temp_tlabel == 1 and class_pred == 0:
						FN[ind] += 1 / np.sum(temp_test_labels)
					
					# Caso 0 -> Positive
					if temp_tlabel == 0 and class_pred == 1:
						FP[ind] += 1 / (temp_test_labels.shape[0]-np.sum(temp_test_labels))
					
			self._prob_pred[temp_test_set == True] = prob_pred_fold
			
		return ({
		  'Accuracy': accuracy,
		  'Sensitivity': sensitivity,
		  'Specificity': specificity,
		  'FP': FP,
		  'FN': FN}, self._prob_pred, models)
	

# ----------------------------------------------------------------------------- #


	def table_kfolds(self, dic: dict):
		jp(pd.DataFrame(
			dic.values(), 
			index = dic.keys(), 
			columns = ['(K: {})'.format(str(i)+' of '+str(len(dic['FP']))) 
			  			for i in range(1,len(dic['FP'])+1)]))


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


	def confusion_matrix(self, dic: dict, table = True):
		confusion_Matrix = np.zeros((2,2))
		confusion_Matrix[0,0] = round(dic['Avg']['Sensitivity'], 6) # TP
		confusion_Matrix[1,0] = round(dic['Avg']['FP'], 6)
		confusion_Matrix[0,1] = round(dic['Avg']['FN'], 6)
		confusion_Matrix[1,1] = round(dic['Avg']['Specificity'], 6) # TN

		if table:
			jp("### Confusion Matrix:")
			jp(pd.DataFrame(
				confusion_Matrix, 
				index = ['Actual P', 'Actual N'], 
				columns = ['Predicted P', 'Predicted N']))
		return confusion_Matrix


# ----------------------------------------------------------------------------- #


	def ROC_curve(self):
		# Calcola la curva ROC per ciascuna fold e calcola l'AUC
		mean_fpr = np.linspace(0, 1, 100)
		tprs = []
		aucs = []
		fpr, tpr, _ = roc_curve(self._labels, self._prob_pred)
		tprs.append(np.interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)

		# Calcola la media e la deviazione standard delle curve ROC e degli AUC
		mean_tpr = np.mean(tprs, axis=0)
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)

		# Disegna la curva ROC media
		plt.figure(figsize=(8, 6))
		plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
		plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)

		# Imposta le etichette e la legenda
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve')
		plt.legend(loc='lower right', fontsize = 'large')
		plt.grid()
		plt.show()
		return {'fpr':mean_fpr, 'tpr':mean_tpr}


# ----------------------------------------------------------------------------- #


	def external_testing(self, models, data, labels):

		accuracy_E = 0.
		sensitivity_E = 0.
		specificity_E = 0.
		FP_E = 0.
		FN_E = 0.

		N_E = len(labels)

		for i in range(N_E):

			data_E = data[i]
			class_pred_E = 0.

			for k in range(len(models)):

				class_pred_temp_E = models[str(k)].predict([data_E])
				class_pred_temp_E = class_pred_temp_E.astype(float)
				class_pred_E += class_pred_temp_E[0] / len(models)

			# Caso 0 -> Negative
			if labels[i] == 0 and class_pred_E < .5:
				accuracy_E += 1 / N_E
				specificity_E += 1 / (N_E-np.sum(labels))

			# Caso 1 -> Positive
			if labels[i] == 1 and class_pred_E >= .5:
				accuracy_E += 1 / N_E
				sensitivity_E += 1 / np.sum(labels)

			# Caso 1 -> Negative
			if labels[i] == 1 and class_pred_E < .5:
				FN_E += 1 / np.sum(labels)

			# Caso 0 -> Positive
			if labels[i] == 0 and class_pred_E >= .5:
				FP_E += 1 / (N_E-np.sum(labels))
		
		return {
			'Accuracy': accuracy_E,
			'Sensitivity': sensitivity_E,
			'Specificity': specificity_E,
			'FP': FP_E,
			'FN': FN_E
			}



# ///////////////////////////////////////////////////////////////////////////// #
# Class for Neural Nerwork
# ///////////////////////////////////////////////////////////////////////////// #

class NN_Class:

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

	def plt_epochs(self, results, idx_sample):    

		def cost_func(x, x0):
			return (x - x0)**2

		plt.figure(figsize=(15,8))

		P = self.Sigmoid(results['container_weights'] @ results['inputs'][idx_sample])
		E = np.ones(len(P)) * results['correct_outputs'][idx_sample]

		plt.subplot(1,2,1)
		x = np.linspace(0, P[0]*1.1, 1000)  
		x0 = results['correct_outputs'][idx_sample]
		plt.grid()
		plt.plot(x, cost_func(x, x0))
		plt.scatter(E[0],0, color='red', linewidth=5)
		plt.plot(P, cost_func(P, E), marker = 'o')
		plt.title(f'Cost function minimization (sample #{idx_sample})')

		plt.subplot(1,2,2)
		plt.grid()
		plt.plot(np.arange(1, results['epochs']+2), 2*(P - E))
		plt.title(f'(sample #{idx_sample})')
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
		plt.figure(figsize=(8, 6))
		plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
		plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)

		# Imposta le etichette e la legenda
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve')
		plt.legend(loc='lower right', fontsize = 'large')
		plt.grid()
		plt.show()
		return {'fpr':mean_fpr, 'tpr':mean_tpr}


# ///////////////////////////////////////////////////////////////////////////// #
# Class for Deep Learning
# ///////////////////////////////////////////////////////////////////////////// #


class DL_Class:

	def __init__(self):
		self._learning_rate = 0.9

# ----------------------------------------------------------------------------- #

	def Sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	
# ----------------------------------------------------------------------------- #

	def ReLU(self, x):
		return np.maximum(0, x)
	
# ----------------------------------------------------------------------------- #

	def Softmax(self, x: np.array):
		# exp = np.exp(x - np.max(x))  # Sottrarre il massimo valore per stabilizzare l'output
		# return exp / np.sum(exp)
		exp = np.exp(x)
		return exp / np.sum(exp)

# ----------------------------------------------------------------------------- #