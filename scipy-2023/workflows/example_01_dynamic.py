"""Flyte Intro: Grid search Dynamic Tasks."""

from typing import List, Tuple, NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from flytekit import task, workflow, dynamic

# Tasks and workflows are reusable across modules
from workflows.example_00_intro import (
    get_data,
    split_data,
    train_model,
    evaluate,
    FEATURES,
    TARGET,
    Hyperparameters,
)


# ✨ Use named tuples to assign semantic value to data types
TuningResults = NamedTuple(
    "TuningResults",
    best_model=LogisticRegression,
    best_val_acc=float,
    train_acc=float,
    test_acc=float,
)


# ⛰ Scaling by splitting work across multiple tasks
@dynamic
def tune_model(
    hyperparam_grid: List[Hyperparameters],
    tune_data: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> Tuple[LogisticRegression, float]:
    """
    🌊 Dynamic workflows compile the execution graph at runtime,
    enabling flexible execution graphs, whose structure you don't
    know ahead of time.
    """
    train_data, val_data = split_data(
        data=tune_data, test_size=val_size, random_state=random_state
    )
    models = [
        train_model(data=train_data, hyperparameters=hp)
        for hp in hyperparam_grid
    ]
    return get_best_model(models=models, val_data=val_data)


@task
def get_best_model(
    models: List[LogisticRegression], val_data: pd.DataFrame
) -> Tuple[LogisticRegression, float]:
    """
    🔻 We implement a "reduce" function that takes the results from the dynamic
    model tuning workflow to find the best model.
    """
    scores = [
        accuracy_score(val_data[TARGET], model.predict(val_data[FEATURES]))
        for model in models
    ]
    best_index = np.array(scores).argmax()
    return models[best_index], float(scores[best_index])


@workflow
def tuning_workflow(
    hyperparam_grid: List[Hyperparameters],
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TuningResults:

    # get and split data
    data = get_data()
    tune_data, test_data = split_data(
        data=data, test_size=test_size, random_state=random_state
    )

    # tune model over hyperparameter grid
    best_model, best_val_acc = tune_model(
        hyperparam_grid=hyperparam_grid,
        tune_data=tune_data,
        val_size=val_size,
        random_state=random_state,
    )

    # evaluate the model
    train_acc = evaluate(model=best_model, data=tune_data)
    test_acc = evaluate(model=best_model, data=test_data)

    # return model with tuning results
    return best_model, best_val_acc, train_acc, test_acc


if __name__ == "__main__":
    hyperparam_grid = [
        Hyperparameters(C=1.0, max_iter=5000),
        Hyperparameters(C=0.1, max_iter=5000),
        Hyperparameters(C=0.01, max_iter=5000),
        Hyperparameters(C=0.001, max_iter=5000),
    ]
    print(f"{tuning_workflow(hyperparam_grid=hyperparam_grid)}")
