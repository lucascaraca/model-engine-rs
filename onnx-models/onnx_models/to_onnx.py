import os

from onnxmltools import (
    convert_sklearn, 
    convert_catboost, 
    convert_lightgbm, 
    convert_xgboost)
from skl2onnx.common.data_types import FloatTensorType

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

CONVERT_MAP = {
    XGBClassifier: convert_xgboost,
    LGBMClassifier: convert_lightgbm,
    CatBoostClassifier: convert_catboost,
    RandomForestClassifier: convert_sklearn
}

def model_to_onnx(model, model_name, path, n_features):
    
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    onnx_func = CONVERT_MAP.get(type(model))
    if onnx_func is None:
        err_msg = (
            f"model is an instance of {type(model)}. "
            f"model must be an instance of {set(CONVERT_MAP.keys())}"
        )
        raise TypeError(err_msg)
    
    onnx_model = onnx_func(
        model, 
        model_name, 
        initial_types=[("features", FloatTensorType([None, n_features]))]
    )

    with open(path, 'wb') as file:
        file.write(onnx_model.SerializeToString())