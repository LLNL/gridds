import os
from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA
import time
import numpy as np
from .base_model import BaseModel

class ARIMA(BaseModel):
    r"""
    arima is parameterized by :math:`p`, the number of autoregressive terms, :math:`d`,
    the number of differences needed for stationarity and :math:`q`,
    the number of lagged forecast errors in the forecasting equation:
     
    .. math::
        \begin{equation}
        \hat{x_t} = \alpha + \beta_1 x_{t-1} +...+ \beta_p x_{t-p} - \theta_1 e_{t-1} - \theta_q e_{t-q}
        \end{equation}

    where :math:`\beta` represent autoregressive terms, :math:`\theta` represent
    moving average terms, and :math:`\alpha` is a bias/intercept term. 
    Potential uses include autoregression, imputation, and fault detection.
    base implementation from
    `statsmodels.tsa.arima.model.arima <https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html>`_

    :param string name: generic name to save and load this model.
    :param int p: number of autoregressive terms :math:`x_{t-1} + ... + x_{t-p}`
    :param int d: differencing factor between autoregressive terms and moving average terms. 
    :param int q: number of moving average terms. :math:`q` in the equation above.

    """
    USES = ['autoregression','impute']

    def __init__(self, name, p=5,d=1,q=0 ):
        super(ARIMA, self).__init__(name)
        self.name = name
        self.p = p
        self.d = d
        self.q = q

    def fit_transform(self,X,**kwargs):        
        print('univariate for now')
        # model=sm_ARIMA(site_data[:,0],order=(1,1,1))
        preds = []
        for X_dim in range(X.shape[1]):
            self.fit(X[:,X_dim])
            predictions = np.expand_dims(self.predict(X[:,X_dim]), axis=-1)
            preds.append(predictions)
        predictions = np.hstack(preds)#np.expand_dims(predictions, axis=-1)
        return predictions

    def fit(self,X, **kwargs):
        self.model= sm_ARIMA(X, order=(self.p, self.d, self.q))
        self.model=self.model.fit()
        # print("ARIMA FIT:" , model_fit.summary())

    
    def predict(self,X, **kwargs):
        # todo: add a check to ensure fit transform procedure is being called
        return self.model.predict()

    def set_autoregression_controls(self, delay, horizon):
        self.p = delay  # AR terms
        self.q = delay # moving average 
        assert horizon == 1, 'cannot use horizon more than 1 for ARIMA'