"""Flyte Intro: Grid search Map Tasks."""

from dataclasses import dataclass, asdict
from typing import List, Tuple, NamedTuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from dataclasses_json import dataclass_json

from flytekit import task, workflow, map_task, Resources
from flytekit.types.structured import StructuredDataset

from workflows.example_intro import (
    get_data,
    split_data,
    evaluate,
    FEATURES,
    TARGET,
    Hyperparameters,
)
from workflows.example_dynamic import get_best_model, TuningResults


@dataclass_json
@dataclass
class TrainArgs:
    """✨ Create a data-type to encapsulate the arguments to a single training run"""

    data: StructuredDataset
    hyperparameters: Hyperparameters


# ⛰ Scaling by provisioning more compute/memory at the task-level
@task(requests=Resources(cpu="2", mem="1Gi"))
def train_model(hyperparameters: Hyperparameters, data: StructuredDataset) -> LogisticRegression:
    """This is a unary task function for our model to make it mappable"""
    data: pd.DataFrame = data.open(pd.DataFrame).all()
    model = LogisticRegression(**asdict(hyperparameters))
    return model.fit(data[FEATURES], data[TARGET])


from functools import partial

@workflow
def tune_model(
    hyperparam_grid: List[Hyperparameters],
    tune_data: StructuredDataset,
    val_size: float,
    random_state: int,
) -> Tuple[LogisticRegression, float]:
    """And finally, a workflow that performs grid search."""
    train_data, val_data = split_data(
        data=tune_data, test_size=val_size, random_state=random_state
    )
    # ⛰ Scaling by splitting work across multiple tasks:
    # Wrapping the `train_model` task in `map_task` allows us to parallelize
    # our grid search.

    partial(train_model, data=train_data)
    models = map_task(partial(train_model, data=train_data), concurrency=5)(
        hyperparameters=hyperparam_grid
    )
    return get_best_model(models=models, val_data=val_data)


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
        Hyperparameters(C=1.0),
        Hyperparameters(C=0.1),
        Hyperparameters(C=0.01),
        Hyperparameters(C=0.001),
    ]
    print(f"{tuning_workflow(hyperparam_grid=hyperparam_grid)}")
