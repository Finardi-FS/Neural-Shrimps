import numpy as np
import pandas as pd
from sklearn.model_selection import KFold # divide il set di dati in k fold (sottogruppi) di dimensioni uguali.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import jupyprint.jupyprint as jp


# ///////////////////////////////////////////////////////////////////////////// #
# Class for Machine Learning 
# ///////////////////////////////////////////////////////////////////////////// #

class MyML:

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