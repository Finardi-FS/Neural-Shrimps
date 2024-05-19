import numpy as np
import pandas as pd
from sklearn.model_selection import KFold # divide il set di dati in k fold (sottogruppi) di dimensioni uguali.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import jupyprint.jupyprint as jp

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
		exp = np.exp(x - np.max(x))
		return exp / np.sum(exp)

# ----------------------------------------------------------------------------- #