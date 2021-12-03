# This allows to import datawig
import os
import sys
import random
import pytest
import datawig
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_hastie_10_2
from sklearn.metrics import classification_report


random.seed(0)
path_root = Path(os.getcwd())
sys.path.append(str(path_root))
warnings.filterwarnings("ignore")


@pytest.fixture
def test_data_frame():

    def get_data(data_fn, noise=3e-1):
        X, y = data_fn(n_samples=10000)
        X = X + np.random.randn(*X.shape) * noise
        test_data_df = pd.DataFrame(np.vstack([X.T, y]).T,
                                    columns=[str(i) for i in range(X.shape[-1] + 1)])
        return test_data_df
    return get_data(make_hastie_10_2)


def test_autogluon_imputer_precision_threshold(test_data_frame):
    X = test_data_frame
    label = X.columns[-1]
    features = [x for x in X.columns if x != label]
    X[label] = X[label].astype(str)
    df_train, df_test = datawig.utils.random_split(X.copy())

    imputer = datawig.AutoGluonImputer(
        input_columns=features,
        output_column=label,
        verbosity=2)

    imputer.fit(train_df=df_train, time_limit=10)

    precisions = []
    for precision_threshold in [0.1, 0.5, 0.9, 0.95, .99]:
        imputed = imputer.predict(df_test[features],
                              precision_threshold=precision_threshold,
                              inplace=True)
        report = classification_report(df_test[label],
                                       imputed[label+"_imputed"].fillna(""),
                                       output_dict=True)
        precisions.append({
            'precision_threshold': precision_threshold,
            'empirical_precision_on_test_set': np.mean([report['-1.0']['precision'],
                                                        report['1.0']['precision']
                                                       ])
        })
    df_precisions = pd.DataFrame(precisions)
    precision_deviations = df_precisions['empirical_precision_on_test_set'] \
                            - df_precisions['precision_threshold'] + 0.01

    assert all(precision_deviations > 0)
