import random
import math

import numpy as np


from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from ctgan import CTGAN
from ForestDiffusion import ForestDiffusionModel
from sklearn.base import BaseEstimator,ClassifierMixin


    

class CVSmoteModel(BaseEstimator,ClassifierMixin):
    """
    CVSmoteModel. It's an estimator and not a oversampling strategy like the others class in this file.
    """

    def __init__(
        self,
        splitter,
        model,
        max_depth=None,
        list_k_max=100,
        list_k_step=10,
        take_all_default_value_k=None,
        sampling_strategy="auto",
    ):
        """_summary_

        Parameters
        ----------
        splitter : sk-learn spliter object (or child)
            _description_
        model : _type_
            _description_
        list_k_max : int, optional
            _description_, by default 100
        list_k_step : int, optional
            _description_, by default 10
        """
        self.max_depth = max_depth
        self.model = model
        self.splitter = splitter
        self.list_k_max = list_k_max  
        self.list_k_step = list_k_step  
        self.estimators_ = [0]  
        self.take_all_default_value_k = take_all_default_value_k
        self.sampling_strategy = sampling_strategy

    def fit(self, X, y, sample_weight=None):
        """
        X and y are numpy arrays
        sample_weight is a numpy array
        """
        self.is_fitted_ = True
        self.model = self.model.set_params(max_depth=self.max_depth)

        self.classes_ = []
        self.n_classes_ = []
        if len(y.shape) == 1: ## One output case
            self.n_outputs_ = 1
            classes_k, _ = np.unique(y, return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        else:
            self.n_outputs_ = y.shape[1]
            for k in range(self.n_outputs_):
                classes_k, _ = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
       

        

        unique, counts = np.unique(y, return_counts=True)
        n_positifs = min(counts)
        print('n_positifs',n_positifs)
        list_k_neighbors = [
            5,
            max(int(0.01 * n_positifs), 1),
            max(int(0.1 * n_positifs), 1),
            max(int(np.sqrt(n_positifs)), 1),
            max(int(0.5 * n_positifs), 1),
            max(int(0.7 * n_positifs), 1),
        ]
        if self.take_all_default_value_k is not None:
            list_k_neighbors = list_k_neighbors[
                : self.take_all_default_value_k
            ]  

        list_k_neighbors.extend(
            list(np.arange(1, self.list_k_max, self.list_k_step, dtype=int))
        )

        best_score = -1
        folds = list(
            self.splitter.split(X, y)
        )  
        for k in list_k_neighbors:
            scores = []
            for train, test in folds:
                new_X, new_y = SMOTE(k_neighbors=k,sampling_strategy=self.sampling_strategy).fit_resample(X[train], y[train])
                self.model.fit(X=new_X, y=new_y, sample_weight=sample_weight)
                scores.append(
                    #roc_auc_score(y[test], self.model.predict_proba(X[test]),multi_class='ovr',average='macro') ## Multiclass case
                    roc_auc_score(y[test], self.model.predict_proba(X[test])[:,1]) 
                )
            if sum(scores) > best_score:
                best_k = k

        new_X, new_y = SMOTE(k_neighbors=best_k,sampling_strategy=self.sampling_strategy).fit_resample(X, y)
        self.model.fit(X=new_X, y=new_y, sample_weight=sample_weight)
        if hasattr(self.model, "estimators_"):
            self.estimators_ = self.model.estimators_

    def predict(self, X):
        """
        predict
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        predict_probas
        """
        return self.model.predict_proba(X)#[:,1] ## [:,1] for GridSeacrh scorer average_precision_score


class MGS(BaseOverSampler):
    """
    MGS : Multivariate Gaussian SMOTE
    This method is depreciated when the covariance matrix of the gaussians distributions contain 0 as eigenvalue.
    We recmmand using MGS2 in such cases.
    """

    def __init__(
        self, K, llambda=1.0, sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        self.random_state = random_state

    def _fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        """

        np.random.seed(self.random_state)

        oversampled_X = X
        oversampled_y = y
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            X_positifs = X[y == class_sample] ## current class

            n_minoritaire = X_positifs.shape[0]
            dimension = X.shape[1]
            neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
            neigh.fit(X_positifs)
            neighbor_by_index = neigh.kneighbors(
                X=X_positifs, n_neighbors=self.K + 1, return_distance=False
            )

            n_synthetic_sample = n_samples 
            new_samples = np.zeros((n_synthetic_sample, dimension))
            
            for i in range(n_synthetic_sample):
                indice = np.random.randint(n_minoritaire)
                # The nearrest neighbor selected for the estimation
                indice_neighbors = neighbor_by_index[indice]
                mu = (1 / (self.K + 1)) * X_positifs[indice_neighbors, :].sum(axis=0)
                sigma = (
                    self.llambda
                    * (1 / (self.K + 1))
                    * (X_positifs[indice_neighbors, :] - mu).T.dot(
                        (X_positifs[indice_neighbors, :] - mu)
                    )
                )

                new_observation = np.random.multivariate_normal(
                    mu, sigma, check_valid="raise"
                ).T
                new_samples[i, :] = new_observation
            
            ## Add the generated samples of the class to the final array
            oversampled_X = np.concatenate((oversampled_X, new_samples), axis=0)
            oversampled_y = np.hstack(
                (oversampled_y,np.full(n_samples, class_sample))
            )
        np.random.seed()

        return oversampled_X, oversampled_y


class MGS2(BaseOverSampler):
    """
    MGS2 : Faster version of MGS using SVD decomposition
    """

    def __init__(self, K, llambda=1.0, sampling_strategy="auto", random_state=None):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        self.random_state = random_state

    def _fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        """

        np.random.seed(self.random_state)

        oversampled_X = X
        oversampled_y = y
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            X_positifs = X[y == class_sample] ## current class

            n_minoritaire = X_positifs.shape[0]
            dimension = X.shape[1]
            neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
            neigh.fit(X_positifs)
            neighbors_by_index = neigh.kneighbors(
                X=X_positifs, n_neighbors=self.K + 1, return_distance=False
            )

            n_synthetic_sample = n_samples

            # computing mu and covariance at once for every minority class points
            all_neighbors = X_positifs[neighbors_by_index.flatten()]
            mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                len(X_positifs), self.K + 1, dimension
            ).sum(axis=1)
            centered_X = X_positifs[neighbors_by_index.flatten()] - np.repeat(
                mus, self.K + 1, axis=0
            )
            centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension)
            covs = (
                self.llambda
                * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X)
                / (self.K + 1)
            )

            # spectral decomposition of all covariances
            eigen_values, eigen_vectors = np.linalg.eigh(covs)  ## long
            eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** 0.5
            As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]

            # sampling all new points
            indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
            new_samples = np.zeros((n_synthetic_sample, dimension))
            for i, central_point in enumerate(indices):
                u = np.random.normal(loc=0, scale=1, size=dimension)
                new_observation = mus[central_point, :] + As[central_point].dot(u)
                new_samples[i, :] = new_observation
            ## Add the generated samples of the class to the final array
            oversampled_X = np.concatenate((oversampled_X, new_samples), axis=0)
            oversampled_y = np.hstack(
                (oversampled_y,np.full(n_samples, class_sample))
            )
        np.random.seed()

        return oversampled_X, oversampled_y


class NoSampling(object):
    """
    None rebalancing strategy class
    """

    def fit_resample(self, X, y):
        """
        X is a numpy array
        y is a numpy array of dimension (n,)
        """
        return X, y

class CtganSampler(BaseOverSampler):
    """
    CTGAN methodology for oversampling minority class.
    """
    def __init__(self,epochs,generator_lr=2e-4,discriminator_lr=2e-4,sampling_strategy="auto", random_state=None):
        """
        """
        super().__init__(sampling_strategy=sampling_strategy)
        #self.ctgan = CTGAN(epochs=epochs,generator_lr=generator_lr,discriminator_lr=discriminator_lr)
        self.epochs = epochs
        self.generator_lr=generator_lr
        self.discriminator_lr = discriminator_lr
        self.random_state = random_state

    def _fit_resample(self, X, y=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        """

        np.random.seed(self.random_state)

        oversampled_X = X
        oversampled_y = y
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            X_positifs = X[y == class_sample] ## current class
            n_minoritaire = X_positifs.shape[0]
            ctgan = CTGAN(epochs=self.epochs,generator_lr=self.generator_lr,discriminator_lr=self.discriminator_lr)
            ctgan.fit(X_positifs)
            new_samples = ctgan.sample(n_samples)
            #print(new_samples)
            
            ## Add the generated samples of the class to the final array
            oversampled_X = np.concatenate((oversampled_X, new_samples), axis=0)
            oversampled_y = np.hstack(
                (oversampled_y,np.full(n_samples, class_sample))
            )
        np.random.seed()

        return oversampled_X, oversampled_y
    
class ForesDiffSampler(BaseOverSampler):
    """
    CTGAN methodology for oversampling minority class.
    """
    def __init__(self,sampling_strategy="auto", random_state=None):
        """
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state

    def _fit_resample(self, X, y=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        """

        np.random.seed(self.random_state)

        oversampled_X = X
        oversampled_y = y
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            X_positifs = X[y == class_sample] ## current class
            n_minoritaire = X_positifs.shape[0]
            forest_model = ForestDiffusionModel(X_positifs, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[], diffusion_type='flow', n_jobs=8)
            new_samples = forest_model.generate(batch_size=n_samples )
            print("*****")
            print('n_samples : ',n_samples)
            print('new_samples.shape : ',new_samples.shape)
            #print(new_samples)
            
            ## Add the generated samples of the class to the final array
            oversampled_X = np.concatenate((oversampled_X, new_samples), axis=0)
            oversampled_y = np.hstack(
                (oversampled_y,np.full(n_samples, class_sample))
            )
        np.random.seed()

        return oversampled_X, oversampled_y
    



class CVBSmoteModel(object):
    """
    CVSmoteModel. It's an estimator and not a oversampling strategy like the others class in this file.
    """

    def __init__(
        self,
        splitter,
        model,
        list_k_max=100,
        list_k_step=10,
        take_all_default_value_k=None,
        sampling_strategy="auto",
        kind = "borderline-1",
    ):
        """_summary_

        Parameters
        ----------
        splitter : sk-learn spliter object (or child)
            _description_
        model : _type_
            _description_
        list_k_max : int, optional
            _description_, by default 100
        list_k_step : int, optional
            _description_, by default 10
        """
        self.splitter = splitter
        self.list_k_max = list_k_max  
        self.list_k_step = list_k_step  
        self.model = model
        self.estimators_ = [0]  
        self.take_all_default_value_k = take_all_default_value_k
        self.sampling_strategy = sampling_strategy
        self.kind = kind

    def fit(self, X, y, sample_weight=None):
        """
        X and y are numpy arrays
        sample_weight is a numpy array
        """

        unique, counts = np.unique(y, return_counts=True)
        n_positifs = min(counts)
        #print('n_positifs',n_positifs)
        list_k_neighbors = [
            5,
            max(int(0.01 * n_positifs), 1),
            max(int(0.1 * n_positifs), 1),
            max(int(np.sqrt(n_positifs)), 1),
            max(int(0.5 * n_positifs), 1),
            max(int(0.7 * n_positifs), 1),
        ]
        if self.take_all_default_value_k is not None:
            list_k_neighbors = list_k_neighbors[
                : self.take_all_default_value_k
            ]  

        list_k_neighbors.extend(
            list(np.arange(1, self.list_k_max, self.list_k_step, dtype=int))
        )

        best_score = -1
        folds = list(
            self.splitter.split(X, y)
        )  
        for k in list_k_neighbors:
            scores = []
            for train, test in folds:
                new_X, new_y = BorderlineSMOTE(k_neighbors=k,sampling_strategy=self.sampling_strategy,kind=self.kind).fit_resample(X[train], y[train])
                self.model.fit(X=new_X, y=new_y, sample_weight=sample_weight)
                scores.append(
                    #roc_auc_score(y[test], self.model.predict_proba(X[test]),multi_class='ovr',average='macro') ## Multiclass case
                    roc_auc_score(y[test], self.model.predict_proba(X[test])[:,1]) 
                )
            if sum(scores) > best_score:
                best_k = k

        new_X, new_y = BorderlineSMOTE(k_neighbors=best_k,sampling_strategy=self.sampling_strategy,kind=self.kind).fit_resample(X, y)
        self.model.fit(X=new_X, y=new_y, sample_weight=sample_weight)
        if hasattr(self.model, "estimators_"):
            self.estimators_ = self.model.estimators_

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
