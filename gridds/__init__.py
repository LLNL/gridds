from . import tools
from . import models
from . import sql

MODEL_NAMES = ['ARIMA', 'BayesianRidge', 'GP', 'KNN', 'LSTM', \
         'VanillaRNN', 'RandomForest', 'VRAE']

def list_model_types(return_string=False):
    res = []
    for model in MODEL_NAMES:
        model = eval("models."+ model)
        if return_string:
            return model
        else:
            res.append(model)
    return res

all = [  
    'tools',
    'models',
    'sql'
    'list_model_types'
        ]   

list_model_types()