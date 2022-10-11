# from .ddcrf import ddcrf
# from .matrix_profile import matrix_profile
# from .lagrange import Lagrange
# from .LSTM_AE import LSTM_AE
from .arima import ARIMA
from .RNN import VanillaRNN
from .LSTM import LSTM
from .VRAE import VRAE
from .KNN import KNN
from .bayesian_ridge import BayesianRidge
from .random_forest import RandomForest
from .gaussian_process import GP
       


all = [ #'matrix_profile',
        # 'ddcrf',
        'ARIMA',
        'VanillaRNN',
        'LSTM',
        'Lagrange',
        'KNN',
        'BayesianRidge',
        'RandomForest',
        'VRAE',
        'GP'

       ]   