import random

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

class MySmote(object):
    """
    MySMOTE. 
    """
    
    def __init__(self, K, w_simulation, w_simulation_params):
        """
        w_simulation is a function.
        w_simulation_params should be a dict corresponding to w_simulation inputs.
        """
        self.K = K
        self.w_simulation = w_simulation
        self.w_simulation_params = w_simulation_params
        
    def fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.
        """

        if y is None:
            X_positifs = X
            X_negatifs = np.ones((0, X.shape[1]))
            assert n_final_sample is not None, "You need to provide a number of final samples."
        else:
            X_positifs = X[y == 1]
            X_negatifs = X[y == 0]
            if n_final_sample is None:
                n_final_sample = (y == 0).sum()
    
        n_minoritaire = X_positifs.shape[0]
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm='ball_tree')
        neigh.fit(X_positifs)
        neighbor_by_index = neigh.kneighbors(X=X_positifs, n_neighbors=self.K+1, return_distance=False) 
        #the first element is always the given point (my nearest neighbor is myself).

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, X.shape[1]))
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire) #central point that is chosen
            k1 = np.random.randint(1,self.K+1)# nearest neighbors which is choisen. The central point itself is excluded (0 excluded)
            indice_neighbor = neighbor_by_index[indice][k1]
            w = self.w_simulation(**self.w_simulation_params) #factor of the difference betweeen the central point and the selected neighbor
            new_samples[i,:] = X_positifs[indice] + w * (X_positifs[indice_neighbor] - X_positifs[indice]) 

        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack((np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1)))

        return oversampled_X, oversampled_y

    
class CVSmoteModel(object):
    """
    CVSmoteModel. It's an estimator and not a oversampling strategy only like the others class in this file.
    """
    
    def __init__(self, splitter, model, list_k_max=100,list_k_step=10):
        """
        splitter is a sk-learn spliter object (or child)
        list_k_max is an integer
        list_k_step is an integer
        """
        self.splitter = splitter
        self.list_k_max = list_k_max
        self.list_k_step = list_k_step
        self.model = model
        self.estimators_= [0]

    def fit(self, X, y, sample_weight=None):
        """
        X and y are numpy arrays
        sample_weight is a numpy array
        """
        
        n_positifs = np.array(y, dtype=bool).sum()
        list_k_neighbors = [5, max(int(0.01*n_positifs),1), max(int(0.1*n_positifs),1), 
                            max(int(np.sqrt(n_positifs)),1)]
        list_k_neighbors.extend(list(
            np.arange(1,self.list_k_max,self.list_k_step,dtype=int)))

        best_score = -1
        folds = list(self.splitter.split(X, y))
        for k in list_k_neighbors:
            scores = []
            for train, test in folds:
                new_X, new_y = SMOTE(k_neighbors=k).fit_resample(X[train], y[train])
                self.model.fit(new_X, new_y, sample_weight)
                scores.append(roc_auc_score(y[test], self.model.predict_proba(X[test])[:,1]))
            if sum(scores) > best_score:
                best_k = k

        new_X, new_y = SMOTE(k_neighbors=best_k).fit_resample(X, y)
        self.model.fit(new_X, new_y, sample_weight)
        if hasattr(self.model, 'estimators_'):
            self.estimators_=self.model.estimators_
        
    
    def predict(self, X):
        """
        predict
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        predict_probas
        """
        return self.model.predict_proba(X)        

class MGS(object):
    """
    MGS
    """
    
    def __init__(self,K,n_points,llambda):
        """
        llambda is a float.
        """
        self.K = K
        self.llambda = llambda
        self.n_points = n_points
        
    def fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.
        """

        if y is None:
            X_positifs = X
            X_negatifs = np.ones((0, X.shape[1]))
            assert n_final_sample is not None, "You need to provide a number of final samples."
        else:
            X_positifs = X[y == 1]
            X_negatifs = X[y == 0]
            if n_final_sample is None:
                n_final_sample = (y == 0).sum()
    
        n_minoritaire = X_positifs.shape[0]
        dimension = X.shape[1]
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm='ball_tree')
        neigh.fit(X_positifs)
        neighbor_by_index = neigh.kneighbors(X=X_positifs, n_neighbors=self.K+1, return_distance=False) 

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire)   
            indices_neigh = [0]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(random.sample(range(1,self.K+1),self.n_points)) # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]
            mu = (1/self.n_points) * X_positifs[indice_neighbors,:].sum(axis=0)
            sigma = self.llambda*(1/self.n_points)* (X_positifs[indice_neighbors,:]-mu).T.dot((X_positifs[indice_neighbors,:]-mu))

            new_observation = np.random.multivariate_normal(mu, sigma, check_valid='raise').T
            new_samples[i,:] = new_observation

        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack((np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1)))

        return oversampled_X, oversampled_y
    
class NoSampling(object):
        
    def fit_resample(self, X, y):
        """
        X is a numpy array
        y is a numpy array of dimension (n,)
        """
        return X,y