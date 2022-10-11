Tasks
==================================

- Tasks are a set of parameters that define an experiment.
- `Autoregression`, `Interpolation`, and    `Fault Detection` all
share the same fields: `procedure`, `name` and `metrics`.
- `Name` is used to identify is the task is autoregression, interpolation, or fault detection.
- `Metrics` are used to identify the means of evaluating performance in a specific task.
    - Must be part of `gridds.tools.metrics.py`.
- `Procedure` is used to define if we are doing `fit` then `predict` or  `fit_transform`.
- `Autoregression` tasks have parameters that define `delay`` and `horizon`.
    - `delay`: number of measurements to be used in history for autoregression.
    - `horizon`: number of measurements to be predicted in future for autoregression.
- Here are some examples::

    default_autoregression = {
        'name': 'autoregression',
        'procedure': ['fit_transform'],
        'metrics': [mae, rmse],
        'delay': 5, 
        'horizon': 1 
    }    

    long_autoregression = {
        'name': 'autoregression',
        'procedure': ['fit_transform'],
        'metrics': [mae, rmse],
        'delay': 20,
        'horizon': 5
    }    


    default_impute = {
        'name': 'impute',
        'procedure': ['fit_transform'],
        'metrics': [mae, rmse]
    }    

    default_fault_pred = {
        'name': 'fault_pred',
        'procedure': ['fit','predict'],
        'metrics': [binary_crossentropy]
    }    

    unsupervised_fault_pred = {
        'name': 'fault_pred',
        'procedure': ['predict'],
        'metrics': [binary_crossentropy]
    }   