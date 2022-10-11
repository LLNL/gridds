from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from .base_model import BaseModel
import numpy as np


class KNN(BaseModel):
    """

    K nearest neighbors wrapper around implementation
    from `sklearn.neighbors.KNeighborsRegressor 
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`_

    :param string name: the name of the method.
    :param int n_neighbors: number of neighbors to use.

    """
    USES = ['autoregression','impute', 'fault_pred']
    def __init__(self, name, n_neighbors=10):
        super(KNN, self).__init__(name)
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        # self.model = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y, **kwargs):
        self.model.fit(X,y)
    
    def predict(self,X,  **kwargs):
        return self.model.predict(X)

    def fit_transform(self,X,  **kwargs):
        self._model = IterativeImputer(random_state=np.random.randint(0,10000),skip_complete=True,estimator=self.model,verbose=False)
        return self._model.fit_transform(X)
    
    def impute(self,X, **kwargs):
        # overrides self.model from init to impure
        self.model = KNNImputer(n_neighbors=self.n_neighbors)
        return self.model.fit_transform(X)

    def transform(self, X,  **kwargs):
        return self.model.transform(X)
