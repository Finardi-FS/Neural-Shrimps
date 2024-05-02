import numpy as np

class PCA:
    def __init__(self, data):
        self._cov_mat = np.corrcoef(data, rowvar=False)
        self._eval, self._evec = np.linalg.eig(self._cov_mat)
    def get_pca(self):
        cov_mat = np.corrcoef(data_int, rowvar=False)        # Calcola matrice correlazione di Pearson tra le features  (check rowvar)
        eig_val, eig_vec = np.linalg.eig(cov_mat)            # Calcola autovettori e autovalori matrice covarianza
sorted_idcs = np.argsort(eig_val)[::-1]

eig_val = eig_val[sorted_idcs]
eig_vec = eig_vec[:,sorted_idcs]
sorted_features = list(np.array(features)[sorted_idcs])
sorted_data_int = data_int[:,sorted_idcs]           # Dataset ordinato in ordine decrescente rispetto agli autovalori
sorted_data_ext = data_ext[:,sorted_idcs]