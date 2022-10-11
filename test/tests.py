import numpy as np
import pytest
from gridds.data import synthetic_data,SmartDS as SmartDS
from gridds import list_model_types
import gridds.tools.tasks as tasks
from gridds.models import ARIMA, VanillaRNN, LSTM,\
     KNN, BayesianRidge, RandomForest, VRAE, GP
# content of conftest.py or a tests file (e.g. in your tests or root directory)

class model_obj:
    def __init__(self, load_ar=True, load_impute=True, load_fault_pred=True) -> None:
        self.load_ar = load_ar
        self.load_impute = load_impute
        self.load_fault_pred = load_fault_pred
        self.ar_models, self.impute_models, self.fault_pred_models= [], [], []
    def load(self):
        for model in list_model_types():
            if 'autoregression' in model.USES and self.load_ar:
                self.ar_models.append(model)
            if 'impute' in model.USES and self.load_impute:
                self.impute_models.append(model)
            if 'fault_pred' in model.USES and self.load_fault_pred:
                self.fault_pred_models.append(model)

@pytest.fixture(scope="session", autouse=True)
def models(request):
    models = model_obj()
    models.load()
    return models


@pytest.fixture(scope="session", autouse=True)
def synthetic_dataset(request):
    synthetic_dataset = synthetic_data.SyntheticData('pytest_dataset')
    reader_instructions = {'features': 1,
                            'sources': 1,
                            'target': 'fault_present'}
    synthetic_dataset.prepare_data(reader_instructions=reader_instructions)
    return synthetic_dataset

@pytest.fixture(scope="session", autouse=True)
def smart_dataset(request):
    smart_dataset = SmartDS('test', sites=1, normalize=False, size=2000)

    reader_instructions = {
        'sources': ['P1U'],
        'modalities': ['load_data'],
        'target': '',  ## how to get faults  for NREL?
        # 'features': ['feature0','feature1', 'feature2', 'feature3']
        'replicates': ['customers']
    }

    smart_dataset.prepare_data(reader_instructions)

    return smart_dataset

@pytest.fixture(scope="session", autouse=True)
def multivar_synthetic_dataset(request):
    multivar_synthetic_dataset = synthetic_data.SyntheticData('pytest_dataset')
    reader_instructions = {'features': 1,
                            'sources': 1,
                            'target': 'fault_present'}
    multivar_synthetic_dataset.prepare_data(reader_instructions=reader_instructions)
    return multivar_synthetic_dataset

def test_prepare_data_int(synthetic_dataset):
    """
    test synthetic data preparation using integers

    TODO: expand dictionary types
    """
    reader_instructions = {'features': 1,
                            'sources': 1,
                            'target': 'fault_present'}
    synthetic_dataset.prepare_data(reader_instructions=reader_instructions)

def test_prepare_data(synthetic_dataset):
    """
    test synthetic data preparation using floats

    """
    reader_instructions = {'features': ['channel_1'],
                            'sources': 1,
                            'target': 'fault_present'}
    synthetic_dataset.prepare_data(reader_instructions=reader_instructions)

def test_shuffle_data(synthetic_dataset):
    """
    test synthetic data preparation using floats

    """
    reader_instructions = {'features': ['channel_1'],
                            'sources': 1,
                            'target': 'fault_present'}
    synthetic_dataset.prepare_data(reader_instructions=reader_instructions)
    synthetic_dataset.shuffle_and_split()

    reader_instructions = {'features': 1,
                            'sources': 1,
                            'target': 'fault_present'}
    synthetic_dataset.prepare_data(reader_instructions=reader_instructions)
    synthetic_dataset.shuffle_and_split()

# def test_multivar_synth(synthetic_dataset):
#     """
#     test synthetic data preparation using multivariate data

#     """
#     reader_instructions = {'features': ['channel_1','channel_2','channel_3'],
#                             'sources': 1,
#                             'target': 'fault_present'}
#     synthetic_dataset.prepare_data(reader_instructions=reader_instructions)
#     # should match # of features we specified
#     assert synthetic_dataset.X.shape[1] == 3
#     synthetic_dataset.shuffle_and_split()

######### AR TESTS ##############

def test_arima_models_ar_synth(synthetic_dataset,models):
    task = tasks.default_autoregression
    model = ARIMA('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_KNN_models_ar_synth(synthetic_dataset,models):
    task = tasks.default_autoregression
    model = KNN('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_VanillaRNN_models_ar_synth(synthetic_dataset,models):
    task = tasks.default_autoregression
    model = VanillaRNN('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_LSTM_models_ar_synth(synthetic_dataset,models):
    task = tasks.default_autoregression
    model = LSTM('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_VRAE_models_ar_synth(synthetic_dataset,models):
    task = tasks.default_autoregression
    model = VRAE('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_BayesianRidge_models_ar_synth(synthetic_dataset,models):
    task = tasks.default_autoregression
    model = BayesianRidge('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)


def test_ar_models_fit_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    for model in models.ar_models:
        model = model('name')
        assert hasattr(model,'fit')
        assert hasattr(model,'predict')


######## FAULT PRED TESTS ##############

def test_KNN_models_fault_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    model = KNN('name')
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_VanillaRNN_models_fault_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    model = VanillaRNN('name')
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_LSTM_models_fault_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    model = LSTM('name')
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_VRAE_models_fault_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    model = VRAE('name')
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)

def test_BayesianRidge_models_fault_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    model = BayesianRidge('name')
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)


def test_models_fault_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    for model in models.fault_pred_models:
        model = model('name')
        assert hasattr(model,'fit')
        assert hasattr(model,'predict')

def test_RF_models_fault_synth(synthetic_dataset,models):
    task = tasks.default_fault_pred
    model = RandomForest('name')
    model.fit(X=synthetic_dataset.X, y=synthetic_dataset.y)


def test_KNN_models_impute_synth(synthetic_dataset,models):
    task = tasks.default_impute
    model = KNN('name')
    model.fit_transform(X=synthetic_dataset.X)


def test_BayesianRidge_models_impute_synth(synthetic_dataset,models):
    task = tasks.default_impute
    model = BayesianRidge('name')
    model.fit_transform(X=synthetic_dataset.X)

def test_GP_models_impute_synth(synthetic_dataset,models):
    task = tasks.default_impute
    model = GP('name')
    # TODO: very inconvienent to require a kwarg to run GP
    model.fit_transform(X=synthetic_dataset.X[:300], timestamps=synthetic_dataset.timestamps[:300])

def test_RF_models_impute_synth(synthetic_dataset,models):
    task = tasks.default_impute
    model = RandomForest('name')
    model.fit_transform(X=synthetic_dataset.X)


def test_models_impute_synth(synthetic_dataset,models):
    task = tasks.default_impute
    for model in models.fault_pred_models:
        model = model('name')
        assert hasattr(model,'fit_transform')


###########################################################################
# NREL Data Tests
############################################################

######### AR TESTS ##############

def test_arima_models_ar_smart_ds(smart_dataset,models):
    task = tasks.default_autoregression
    model = ARIMA('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_KNN_models_ar_smart_ds(smart_dataset,models):
    task = tasks.default_autoregression
    model = KNN('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_VanillaRNN_models_ar_smart_ds(smart_dataset,models):
    task = tasks.default_autoregression
    model = VanillaRNN('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_LSTM_models_ar_smart_ds(smart_dataset,models):
    task = tasks.default_autoregression
    model = LSTM('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_VRAE_models_ar_smart_ds(smart_dataset,models):
    task = tasks.default_autoregression
    model = VRAE('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_BayesianRidge_models_ar_smart_ds(smart_dataset,models):
    task = tasks.default_autoregression
    model = BayesianRidge('name')
    model.set_autoregression_controls(task['delay'], task['horizon'])
    model.fit(X=smart_dataset.X, y=smart_dataset.y)


def test_ar_models_fit_synth(smart_dataset,models):
    task = tasks.default_fault_pred
    for model in models.ar_models:
        model = model('name')
        assert hasattr(model,'fit')
        assert hasattr(model,'predict')


######## FAULT PRED TESTS ##############

def test_KNN_models_fault_smart(smart_dataset,models):
    task = tasks.default_fault_pred
    model = KNN('name')
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_VanillaRNN_models_fault_smart(smart_dataset,models):
    task = tasks.default_fault_pred
    model = VanillaRNN('name')
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_LSTM_models_fault_smart(smart_dataset,models):
    task = tasks.default_fault_pred
    model = LSTM('name')
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_VRAE_models_fault_smart(smart_dataset,models):
    task = tasks.default_fault_pred
    model = VRAE('name')
    model.fit(X=smart_dataset.X, y=smart_dataset.y)

def test_BayesianRidge_models_fault_smart(smart_dataset,models):
    task = tasks.default_fault_pred
    model = BayesianRidge('name')
    model.fit(X=smart_dataset.X, y=smart_dataset.y)


def test_models_fault_smart(smart_dataset,models):
    task = tasks.default_fault_pred
    for model in models.fault_pred_models:
        model = model('name')
        assert hasattr(model,'fit')
        assert hasattr(model,'predict')

def test_RF_models_fault_smart(smart_dataset,models):
    task = tasks.default_fault_pred
    model = RandomForest('name')
    model.fit(X=smart_dataset.X, y=smart_dataset.y)


def test_KNN_models_impute_smart(smart_dataset,models):
    task = tasks.default_impute
    model = KNN('name')
    model.fit_transform(X=smart_dataset.X)


def test_BayesianRidge_models_impute_smart(smart_dataset,models):
    task = tasks.default_impute
    model = BayesianRidge('name')
    model.fit_transform(X=smart_dataset.X)

# def test_GP_models_impute_smart(smart_dataset,models):
#     task = tasks.default_impute
#     model = GP('name')
#     # TODO: very inconvienent to require a kwarg to run GP
#     model.fit_transform(X=smart_dataset.X[:300], timestamps=smart_dataset.timestamps[:300])

def test_RF_models_impute_smart(smart_dataset,models):
    task = tasks.default_impute
    model = RandomForest('name')
    model.fit_transform(X=smart_dataset.X)


def test_models_impute_smart(smart_dataset,models):
    task = tasks.default_impute
    for model in models.fault_pred_models:
        model = model('name')
        assert hasattr(model,'fit_transform')


# test case
# ami_df[ami_df['start_time'] == pd.to_datetime('2019-07-30 20:00:00')]['start_time'].values <  oms_df[oms_df['customers_affected'] == '583']['start_time'].values
    
# ami_df[ami_df['start_time'] == pd.to_datetime('2019-07-30 20:00:00')]['end_time'].values >  oms_df[oms_df['customers_affected'] == '583']['end_time'].values

#lagged_timeseries[20] == time_series[0] for delay 20


# test if VAE can handle several different lengths to be robust