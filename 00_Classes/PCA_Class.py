import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jupyprint.jupyprint as jp

# ///////////////////////////////////////////////////////////////////////////// #
# Class for Principal Component Analysis (PCA)
# ///////////////////////////////////////////////////////////////////////////// #

class MyPCA:

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