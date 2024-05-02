import numpy as np
from scipy.io import loadmat
import pandas as pd
from jupyprint import jupyprint, arraytex
from sklearn.model_selection import KFold # divide il set di dati in k fold (sottogruppi) di dimensioni uguali.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt




class ABC123:
	def __init__(self, data, labels):
		self._data = data
		self._labels = labels
		self._numVar = self._data.shape[1]
		self._N = len(self._labels)
		self._original_indices = np.arange(self._N)
		

# -------------------------------------------------------------------------------------------------------------------- #


	def get_data(self):
		return {'data': self._data, 'labels': self._labels}    


# -------------------------------------------------------------------------------------------------------------------- #


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
			jupyprint('### Indici delle fold associate a ciascun campione:')
			jupyprint(indices)
		return (K, indices)
