
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from .base_model import BaseModel
from sklearn.linear_model import BayesianRidge as BRIDGE


class BayesianRidge(BaseModel):
    """
    Bayesian form of ridge regression. 
    Assumes a normally distributed output 
    $\bm{y_t}$ with linear relationship to inputs 
    $\bm{x_t}$, formulated as 
    $\bm{y_t} = \bm{x_t}\bm{w} + \bm{\alpha}$ 
    with likelihood $p(\bm{y_t} \mid \bm{x_t}, \bm{w}, \bm{\alpha})$ 
    where $ \bm{\alpha}$ are continuous random variables 
    representing bias estimated from $X$. 
    Weights are sampled from $\mathcal{N}(\bm{x_t}\bm{w},\bm{\alpha})$ 
    and the prior for $\bm{w}$ is given by the spherical Gaussian. 
    The coefficients, $\bm{w}$ are optimized by Markov Chain 
    Monte Carlo sampling. 

    TODO: add ref for \citep{mackay1992bayesian}
    :param str name: the name attribute of the method.

    """
    def __init__(self, name):  # , full_name):
        self.name = name
        super(BayesianRidge, self).__init__(name)

        # super(BayesianRidge)
        state=np.random.randint(0,1000)
        est = BRIDGE()
        self.model = IterativeImputer(estimator=est,\
            random_state=state,sample_posterior=False,verbose=False)
        #self.model = IterativeImputer(random_state=0, verbose=False)

    def fit(self,X, **kwargs):
        self.model.fit(X)

    def predict(self,X, **kwargs):
        return self.model.predict(X)

    def fit_transform(self,X, **kwargs):
        return self.model.fit_transform(X)

    def impute(self,X, **kwargs):
        return self.model.fit_transform(X.reshape(-1,1))

    def transform(self,X, **kwargs):
        return self.model.transform(X)


