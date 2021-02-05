import logging

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.results import save_predictions
from amlb.utils import Timer

from sklearn.preprocessing import OrdinalEncoder

import numpy as np
import pandas as pd

from frameworks.shared.callee import save_metadata

import torch

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

log = logging.getLogger(__name__)

def run(dataset:Dataset, config: TaskConfig):
    log.info("****TabNet****")
    save_metadata(config)

    is_classification = config.type == 'classification'
    X_train, X_test = dataset.train.X, dataset.test.X
    X_train, X_test = impute(dataset.train.X, dataset.test.X, strategy="most_frequent")

    X = np.concatenate((X_train, X_test), axis=0)
    enc = OrdinalEncoder()
    enc.fit(X)
    X_train = enc.transform(X_train)
    X_test = enc.transform(X_test)

    y_train, y_test = dataset.train.y, dataset.test.y

    estimator = TabNetClassifier if is_classification else TabNetRegressor
    predictor = estimator() # дефолтный табнет ничего не просит, но можно тут регулировать метапараметры

    # if not is_classification:
    #     y_train = torch.reshape(torch.from_numpy(y_train), (-1, 1))
    #     y_test = torch.reshape(torch.from_numpy(y_test), (-1, 1))

    if not is_classification:
        y_train = np.reshape(y_train.astype(np.float32), (-1, 1))
        y_test = np.reshape(y_test.astype(np.float32), (-1, 1))

    # if not is_classification:
    #     y_train = torch.reshape(torch.from_numpy(y_train.astype(float)), (-1, 1))
    #     y_test = torch.reshape(torch.from_numpy(y_test.astype(float)), (-1, 1))

    with Timer() as training:
        predictor.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)])
    with Timer() as predict:
        predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test)
    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration
    )
