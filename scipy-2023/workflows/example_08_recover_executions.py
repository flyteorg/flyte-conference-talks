"""Recoverability: Recover from executions."""

from dataclasses import asdict
from random import random
from typing import List

import pandas as pd

from sklearn.linear_model import SGDClassifier

from flytekit import task, workflow, dynamic
from flytekit.exceptions.user import FlyteRecoverableException

from workflows.example_07_caching import get_best_model
from workflows.example_06_reproducibility import (
    Hyperparameters,
    FEATURES,
    TARGET,
)
from workflows.example_07_caching import get_data, split_data


FAILURE_RATE = 0.25


@task(cache=True, cache_version="2", retries=3)
def train_model(
    data: pd.DataFrame, hyperparameters: Hyperparameters
) -> SGDClassifier:
    """
    🎒 Caching and workflow recovery allows you to recover from a grid search tuning
    workflow that may have failed due to system-level or exogenous factors.

    Combined with retries 🔂, this makes your workflows robust to uncontrollable
    weirdness in the world 🌍.
    """
    print(f"training with hyperparameters: {hyperparameters}")

    # simulate system-level error: per trail, introduce
    # a chance of failure 25% of the time
    if random() < FAILURE_RATE:
        raise FlyteRecoverableException(
            f"🔥 Something went wrong with hyperparameters {hyperparameters}! 🔥"
        )

    return SGDClassifier(**asdict(hyperparameters)).fit(
        data[FEATURES], data[TARGET]
    )


@dynamic
def tune_model(
    alpha_grid: List[float],
    tune_data: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> SGDClassifier:

    hyperparam_grid = [Hyperparameters(alpha=alpha) for alpha in alpha_grid]
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
    alpha_grid: List[float],
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
        alpha_grid=alpha_grid,
        tune_data=tune_data,
        val_size=val_size,
        random_state=random_state,
    )
    return best_model


if __name__ == "__main__":
    alpha_grid = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    print(f"{tuning_workflow(alpha_grid=alpha_grid)}")
