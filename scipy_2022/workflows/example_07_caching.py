"""Recoverability: Caching for compute efficiency."""

from dataclasses import asdict
from typing import List

import pandas as pd

from sklearn.linear_model import SGDClassifier

from flytekit import task, workflow, dynamic


from workflows.example_00_intro import split_data
from workflows.example_01_dynamic import get_best_model
from workflows.example_06_reproducibility import (
    get_data,
    Hyperparameters,
    FEATURES,
    TARGET,
)


@task(cache=True, cache_version="1")
def train_model(
    data: pd.DataFrame, hyperparameters: Hyperparameters
) -> SGDClassifier:
    """
    🎒 Caching allows you to recover from a grid search tuning workflow so that you
    don't have to re-train models given the same data and hyperparameters.
    """
    print(f"training with hyperparameters: {hyperparameters}")
    return SGDClassifier(**asdict(hyperparameters)).fit(
        data[FEATURES], data[TARGET]
    )


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
        train_model(data=train_data, hyperparameters=hp)
        for hp in hyperparam_grid
    ]
    model, _ = get_best_model(models=models, val_data=val_data)
    return model


@workflow
def tuning_workflow(
    hyperparam_grid: List[Hyperparameters],
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SGDClassifier:

    # get and split data
    data = get_data()
    tune_data, _ = split_data(
        data=data, test_size=test_size, random_state=random_state
    )

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
        Hyperparameters(alpha=alpha)
        for alpha in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
    ]
    print(f"{tuning_workflow(hyperparam_grid=hyperparam_grid)}")
