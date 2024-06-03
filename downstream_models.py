import numpy as np

from implicit_slim import implicit_slim, slim_lle_1, slim_lle_2

from scipy.linalg import orth
from sklearn.utils.extmath import randomized_svd


class MF:
    def __init__(self, X, params):
        n_users, n_items = X.shape
        
        self.params = params
        self.X = X
        self.Q = np.random.randn(params['L'], n_items)
        self.b = np.array(X.mean(0)).T if params['bias'] else np.zeros((n_items, 1))
        self.current_epoch = 0
                
    def get_P(self, X_test=None):
        r_p = self.params['r_p']
        L = self.params['L']
        X, Q, b = self.X if X_test is None else X_test, self.Q, self.b
        return np.linalg.inv(r_p * np.eye(L) + Q @ Q.T) @ (Q @ X.T - Q @ b)

    def update_P(self):
        self.P = self.get_P()
        
    def get_Q(self):
        if self.params['reg'] == 'none':
            return self.get_Q_vanila()
        elif self.params['reg'] == 'ImplicitSLIM':
            return self.get_Q_regularized()
        else:
            raise
            
    def get_Q_vanila(self):
        r_q = self.params['r_q']
        L = self.params['L']
        X, P, b = self.X, self.P, self.b
        return np.linalg.inv(r_q * np.eye(L) + P @ P.T) @ (P @ X - P.sum(1, keepdims=True) @ b.T)
        
    def update_Q(self):
        self.Q = self.get_Q()
        
    def get_Q_regularized(self):
        r_q = self.params['r_q']
        s_q = self.params['s_q']
        L = self.params['L']
        X, P, V, b = self.X, self.P, self.V, self.b
        return np.linalg.inv((r_q + s_q) * np.eye(L) + P @ P.T) @ (P @ X - P.sum(1, keepdims=True) @ b.T + s_q * V)
        
    def get_SVD_embeddings(self):
        _, _, W = randomized_svd(self.X, n_components=self.params['L'], n_iter=4,
                                 power_iteration_normalizer='QR'
                                )
        return W
    
    def get_LLE_SLIM_embeddings(self):
        λ = self.params['item_λ']
        L = self.params['L']
        B = slim_lle_1(self.X, λ)
        res = slim_lle_2(B, L).T
        return res
        
    def get_ImplicitSLIM_embeddings(self, Q):
        V = implicit_slim(Q, self.X, 
                          self.params['item_λ'], self.params['item_α'], self.params['item_thr']
                         )
        if self.params['orth']:
            V_orth = orth(V.T).T
            V = V_orth if V.shape[0] == V_orth.shape[0] else V
        return V

    def predict(self, X_test):
        P = self.get_P(X_test)
        return P.T @ self.Q + self.b.T
    
    def init(self):
        if self.current_epoch == 0:
            if self.params['init'] == 'none':
                pass
            elif self.params['init'] == 'ImplicitSLIM':
                self.Q = self.get_ImplicitSLIM_embeddings(self.Q)
            elif self.params['init'] == 'SVD':
                self.Q = self.get_SVD_embeddings()
            elif self.params['init'] == 'LLE-SLIM':
                self.Q = self.get_LLE_SLIM_embeddings()
            else:
                raise

    def step(self):
        if self.params['reg'] == 'none':
            pass
        elif self.params['reg'] == 'ImplicitSLIM':
            if self.params['init'] == 'ImplicitSLIM' and self.current_epoch == 0:
                self.V = self.Q.copy()
            else:
                self.V = self.get_ImplicitSLIM_embeddings(self.Q)
        else:
            raise
        
        self.update_P()
        self.update_Q()
        
        self.current_epoch += 1


class PLRec(MF):
    def __init__(self, X, params):
        super().__init__(X, params)
        self.W = np.random.randn(params['L'], n_items)
        self.V = self.W.copy()
        
    def get_P(self, X_test=None):
        X = self.X if X_test is None else X_test
        D = X.sum(0)
        D = np.array(D).T
        D = 1 / (np.power(D, 1 / self.params['power']) + 0.0001)
        return (X @ (D * self.W.T)).T
    
    def init(self):
        if self.params['init'] == 'none':
            pass
        elif self.params['init'] == 'ImplicitSLIM':
            self.W = self.get_ImplicitSLIM_embeddings(self.W)
        elif self.params['init'] == 'SVD':
            self.W = self.get_SVD_embeddings()
            self.V = self.W.copy()
        elif self.params['init'] == 'LLE-SLIM':
            self.W = self.get_LLE_SLIM_embeddings()
        else:
            raise
            
    def step(self):
        if self.params['reg'] == 'none':
            pass
        elif self.params['reg'] == 'ImplicitSLIM':
            if self.params['init'] == 'ImplicitSLIM':
                self.V = self.W
            else:
                self.V = self.get_ImplicitSLIM_embeddings(self.V)
        else:
            raise
            
        self.update_P()
        self.update_Q()
        
        self.current_epoch += 1
        