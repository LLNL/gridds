# GridDS: Data Science Toolkit for Energy Grid Data (GridDS), Version 1.0

Author(s): Alexander Ladd & Indrasis Chakraborty


## Installation
- Python Env
    - First  `pip install -r requirements.txt`
    - Then pip install -e .
- Conda Env
    - `conda install gridds.yml`
    
    
## Quick start
- See `examples/simple_autoregression` for an introduction on how to run **autoregression**
- See `examples/simple_interpolate` for an introduction on how to run **interpolation**
- See `examples/simple_fault_detect` for an introduction on how to run **fault detection**

```
├── RNN_univariate_nrel.db
├── __init__.py
├── data
│   ├── __init__.py
│   ├── archive
│   ├── base_dataset.py
│   ├── cleaners.py
│   ├── csvs
│   ├── db_interface.py
│   ├── install_nrel.sh
│   ├── nrel_smart_ds
│   ├── nrel_smart_ds.py
│   ├── nrel_smart_ds_real
│   ├── saved_datasets
│   └── synthetic_data.py
├── data_specification.py
├── docs
│   ├── Makefile
│   ├── build
│   ├── make.bat
│   └── source
├── examples
│   └── simple_autoregression.ipynb
│   └── simple_forecasting.ipynb
│   └── simple_interpolation.ipynb
├── experimenter.py
├── experiments
│   ├── add_faults.py
│   ├── amlies22
│   ├── ar.sh
│   ├── archive
│   ├── autoregression_basic.py
│   ├── autoregression_basic_clean.py
│   ├── clean_data.py
│   ├── clean_data_clean copy.py
│   ├── clean_data_clean.py
│   ├── composite_experiment.py
│   ├── ingest_data.py
│   ├── ingest_nreca_data.py
│   ├── interpolate_forecast.py
│   ├── launch.sh
│   ├── launch_reponse.sh
│   ├── read_data.py
│   ├── read_data_scada.py
│   ├── read_data_scada_clean.py
│   ├── scada_interpolate.py
│   ├── tune_hyperparams.py
│   └── visualize.py
├── figures
├── hparams.md
├── ingest_data.sh
├── methods
│   ├── KNN.py
│   ├── LSTM.py
│   ├── LSTM_AE.py
│   ├── RNN.py
│   ├── VAE.py
│   ├── XGBoost.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── arima.py
│   ├── base_model.py
│   ├── bayesian_ridge.py
│   ├── ddcrf.py
│   ├── gaussian_process.py
│   ├── lagrange.py
│   ├── matrix_profile.py
│   └── random_forest.py
├── requirements.txt
├── setup.py
├── sql
│   ├── __pycache__
│   ├── ami.sql
│   ├── ami_blink.sql
│   ├── archive
│   ├── drop_tables.sql
│   ├── gis.sql
│   ├── ingest.py
│   ├── notes.md
│   ├── oms.sql
│   ├── query_scratchpad.txt
│   └── scada.sql
├── test
│   └── tests.py
└── tools
    ├── __pycache__
    ├── config.py
    ├── hp_optimization.py
    ├── metrics.py
    ├── private.py
    ├── style.rc
    ├── tasks.py
    ├── utils.py
    ├── viz.py
    └── viz_extension.py

```



### CP NUMBER: CP02594
