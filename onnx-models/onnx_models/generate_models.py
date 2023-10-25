# arrays e dataframes
import numpy as np
import pandas as pd

# modelos
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# geracao de datasets
from sklearn.datasets import make_classification

# funções auxiliares
import gc
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


def get_feature_names(n_features: int):
    
    zfill = len(str(n_features))
    feature_names = [
        f'vp_{str(i).zfill(zfill)}'
        for i in range(1, n_features+1)
    ]

    return feature_names


def generate_random_train_test_dataset(
        n_samples: int,
        n_features: int,
        seed: int = None,
        test_size: float = 0.25
) -> Tuple['pd.DataFrame', 'pd.DataFrame']:

    feature_names = get_feature_names(n_features)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=3,
        random_state=seed
    )

    data = pd.DataFrame(X, columns=feature_names).astype(float)
    data['target'] = y

    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=seed)

    del X, y, data
    gc.collect()

    return train_data, test_data


def get_random_features_map(
    features: List[str],
    min_features: List[str] = 1,
    max_features: List[str] = 1,
    seed: int = None,
    n_experiments: int = 1
) -> Dict[int, List[str]]:

    np.random.seed(seed)

    random_features_map = {}
    for i in range(n_experiments):
        n_features = np.random.randint(low=min_features, high=max_features+1)
        random_features_map[i+1] = sorted(
            np.random.choice(features, size=n_features, replace=False)
        )

    return random_features_map


def get_estimators_map(seed: int = None) -> Dict[str, 'BaseEstimator']:

    estimators_map = {
        'lgbm': LGBMClassifier(random_state=seed,
                               early_stopping_rounds=50,
                               n_estimators=800,
                               max_depth=7,
                               min_split_gain=0.01,
                               verbosity=-1),
        'ctb': CatBoostClassifier(random_state=seed,
                                  early_stopping_rounds=50,
                                  n_estimators=800,
                                  max_depth=7,
                                  verbose=0,
                                  allow_writing_files=False),
        'xgb': XGBClassifier(random_state=seed,
                             early_stopping_rounds=50,
                             max_depth=7,
                             gamma=0.08, verbose=0),
        'rf': RandomForestClassifier(n_estimators=400,
                                     max_depth=7,
                                     random_state=seed,
                                     min_impurity_decrease=0.01)
    }

    return estimators_map
