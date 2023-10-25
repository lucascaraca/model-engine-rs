import os
import json
import argparse

from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import clone as sklearn_clone

from onnx_models.generate_models import (
    get_feature_names,
    generate_random_train_test_dataset,
    get_random_features_map,
    get_estimators_map,
)
from onnx_models.to_onnx import model_to_onnx


def generate_onnx_models(
    seed: int = 42,
    n_samples: int = 30_000,
    n_features: int = 150,
    test_size=0.25,
    n_experiments_per_model: int = 3,
    min_features: int = 60,
    max_features: int = 95,
    export_folder: str = './exported_models',
):
    #
    export_folder = os.path.abspath(export_folder)
    feature_names = get_feature_names(n_features)

    # get train, test data
    train_data, test_data = generate_random_train_test_dataset(
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
        test_size=test_size
    )

    # get feature selection experiments
    feature_experiments = get_random_features_map(
        features=feature_names,
        min_features=min_features,
        max_features=max_features,
        seed=seed,
        n_experiments=n_experiments_per_model
    )

    # get estimators
    estimators_map = get_estimators_map(seed)
    target = 'target'

    for estimator_name, estimator in tqdm(estimators_map.items(), desc="Generating models..."):
        for experiment, features in feature_experiments.items():
            # cria uma nova instancia do modelo
            model = sklearn_clone(estimator)
            model_name = f'{estimator_name}-v{experiment}'

            # definindo alguns parâmetros que acelaram o treino
            fit_kwargs = {}
            if isinstance(estimator, (LGBMClassifier, CatBoostClassifier, XGBClassifier)):
                fit_kwargs['eval_set'] = [
                    (test_data[features], test_data[target])]
            if isinstance(estimator, XGBClassifier):
                fit_kwargs['verbose'] = 0

            # treinando o modelo
            model = model.fit(train_data[features],
                              train_data[target], **fit_kwargs)
            
            # ajuste do XGB para que a serialização seja possível
            if isinstance(estimator, XGBClassifier):
                model.get_booster().feature_names = [
                    f'f{i}' for i in range(len(features))]

            # escorando a base de teste com o modelo
            test_data[model_name] = model.predict_proba(
                test_data[features].to_numpy())[:, 1]
            
            # salvando o modelo
            model_folder = os.path.join(export_folder, estimator_name)
            os.makedirs(model_folder, exist_ok=True)

            model_path = os.path.join(model_folder, f'{model_name}.onnx')
            model_to_onnx(model, model_name, model_path, len(features))

            # salvando metadados do modelo
            metadata = {
                'model_name': model_name,
                'features': features,
            }
            metadata_path = os.path.join(model_folder, f'{model_name}.json')
            with open(metadata_path, 'w') as file:
                json.dump(metadata, file, indent=4)
            
    
    data_path = os.path.join(export_folder, 'val_data.parquet.gzip')
    test_data.to_parquet(data_path, compression='gzip')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Model Generator',
        description='Generates BVS-like Random ML Models'
    )

    parser.add_argument('--export_folder', required=True, help='Folder to export models')
    parser.add_argument('--seed', default=42, help='Sets random seed to all random processes')
    parser.add_argument('--n_samples', default=30_000, help='Number of dataset samples')
    parser.add_argument('--test_size', default=0.25, help='Percentage of dataset n_samples to get test dataset')
    parser.add_argument('--n_exp', default=3, help='Number of experiments for each trained model')
    parser.add_argument('--n_features', 
                        default=150, 
                        help='Number of random features to generate in the dataset')
    parser.add_argument('--min_features', default=42, help='Min number of features to be sorted')
    parser.add_argument('--max_features', default=95, help='Max number of features to be sorted')

    
    

    args = parser.parse_args()
    generate_onnx_models(
        seed=args.seed,
        n_samples=args.n_samples,
        n_features=args.n_features,
        test_size=args.test_size,
        n_experiments_per_model=args.n_exp,
        min_features=args.min_features,
        max_features=args.max_features,
        export_folder=args.export_folder
        )
