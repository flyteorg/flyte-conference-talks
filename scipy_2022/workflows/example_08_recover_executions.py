"""Recover from executions."""

# - example where a task can fail with 10% probability (don't set random seed).
#   - run the task 100 times to expect 10 failures
#   - recover from failures, expect 1 to fail.

from dataclasses import asdict
from random import random
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


@task(cache=True, cache_version="1", retries=3)
def train_model(data: pd.DataFrame, hyperparameters: Hyperparameters) -> SGDClassifier:
    print(f"training with hyperparameters: {hyperparameters}")
    # simulate system-level error: per trail, introduce
    # a chance of failure 5% of the time
    if random() < 0.05:
        raise RuntimeError("🔥 Something went wrong! 🔥")
    model = SGDClassifier(**asdict(hyperparameters))
    return model.fit(data[FEATURES], data[TARGET])


@dynamic
def tune_model(
    n_alpha_samples: int,
    tune_data: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> SGDClassifier:
    hyperparam_grid = [Hyperparameters(alpha=alpha) for alpha in np.geomspace(1e-6, 1e3, n_alpha_samples)]
    train_data, val_data = split_data(data=tune_data, test_size=val_size, random_state=random_state)
    models = [train_model(data=train_data, hyperparameters=hp) for hp in hyperparam_grid]
    model, _ = get_best_model(models=models, val_data=val_data)
    return model


@workflow
def tuning_workflow(
    n_alpha_samples: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SGDClassifier:

    # get and split data
    data = get_data()
    tune_data, _ = split_data(data=data, test_size=test_size, random_state=random_state)
    
    # tune model over hyperparameter grid
    best_model = tune_model(
        n_alpha_samples=n_alpha_samples,
        tune_data=tune_data,
        val_size=val_size,
        random_state=random_state,
    )
    return best_model


if __name__ == "__main__":
    import numpy as np


    print(f"{tuning_workflow(n_alpha_samples=100)}")
