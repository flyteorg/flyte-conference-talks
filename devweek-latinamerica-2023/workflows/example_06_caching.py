"""Recoverability: Caching for compute efficiency."""

from dataclasses import asdict
from typing import Annotated, List, Tuple

import numpy as np
import pandas as pd
from flytekit import HashMethod, ImageSpec, Resources, dynamic, task, workflow
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from workflows.example_05_reproducibility import FEATURES, TARGET, Hyperparameters

custom_image = ImageSpec(registry="samhitaalla", packages=["palmerpenguins"])

if custom_image.is_container():
    from palmerpenguins import load_penguins


@task(container_image=custom_image)
def get_data() -> pd.DataFrame:
    return load_penguins()[[TARGET] + FEATURES].dropna()


def hash_pandas_dataframe(df: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(df))


CachedDataFrame = Annotated[pd.DataFrame, HashMethod(hash_pandas_dataframe)]


@task(cache=True, cache_version="1")
def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[CachedDataFrame, CachedDataFrame]:
    return train_test_split(data, test_size=test_size, random_state=random_state)


@task(
    cache=True,
    cache_version="1",
    retries=3,
    requests=Resources(cpu="2", mem="1Gi"),
    limits=Resources(cpu="2", mem="1Gi"),
)
def train_model(data: pd.DataFrame, hyperparameters: Hyperparameters) -> SGDClassifier:
    print(f"training with hyperparameters: {hyperparameters}")
    return SGDClassifier(**asdict(hyperparameters)).fit(data[FEATURES], data[TARGET])


@task
def get_best_model(
    models: List[SGDClassifier], val_data: pd.DataFrame
) -> Tuple[SGDClassifier, float]:
    scores = [
        accuracy_score(val_data[TARGET], model.predict(val_data[FEATURES]))
        for model in models
    ]
    best_index = np.array(scores).argmax()
    return models[best_index], float(scores[best_index])


@dynamic
def tune_model(
    hyperparam_grid: List[Hyperparameters],
    tune_data: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> SGDClassifier:
    train_data, val_data = split_data(
        data=tune_data, test_size=val_size, random_state=random_state
    )
    models = [
        train_model(data=train_data, hyperparameters=hp) for hp in hyperparam_grid
    ]
    model, _ = get_best_model(models=models, val_data=val_data)
    return model


@workflow
def tuning_workflow(
    hyperparam_grid: List[Hyperparameters] = [
        Hyperparameters(alpha=10.0),
        Hyperparameters(alpha=1.0),
        Hyperparameters(alpha=0.1),
        Hyperparameters(alpha=0.01),
        Hyperparameters(alpha=0.001),
        Hyperparameters(alpha=0.0001),
    ],
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SGDClassifier:
    # get and split data
    data = get_data()
    tune_data, _ = split_data(data=data, test_size=test_size, random_state=random_state)

    # tune model over hyperparameter grid
    best_model = tune_model(
        hyperparam_grid=hyperparam_grid,
        tune_data=tune_data,
        val_size=val_size,
        random_state=random_state,
    )
    return best_model


if __name__ == "__main__":
    hyperparam_grid = [
        Hyperparameters(alpha=alpha) for alpha in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
    ]
    print(f"{tuning_workflow(hyperparam_grid=hyperparam_grid)}")
