"""Grid search map_task example."""

from dataclasses import dataclass
from typing import List, Tuple, NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataclasses_json import dataclass_json

from flytekit import task, workflow, map_task
from flytekit.types.structured import StructuredDataset

from workflows.example_00_intro import (
    get_data,
    split_data,
    evaluate,
    FEATURES,
    TARGET,
)


@dataclass_json
@dataclass
class TrainArgs:
    data: StructuredDataset
    hyperparameters: dict


@task
def prepare_train_args(train_data: StructuredDataset, hyperparam_grid: List[dict]) -> List[TrainArgs]:
    return [TrainArgs(train_data, hp) for hp in hyperparam_grid]


@task
def train_model(train_args: TrainArgs) -> LogisticRegression:
    data: pd.DataFrame = train_args.data.open(pd.DataFrame).all()
    model = LogisticRegression(max_iter=5000, **train_args.hyperparameters)
    return model.fit(data[FEATURES], data[TARGET])


@task
def get_best_model(models: List[LogisticRegression], val_data: StructuredDataset) -> Tuple[LogisticRegression, float]:
    val_data: pd.DataFrame = val_data.open(pd.DataFrame).all()
    scores = [accuracy_score(val_data[TARGET], model.predict(val_data[FEATURES])) for model in models]
    best_index = np.array(scores).argmax()
    return models[best_index], float(scores[best_index])


@workflow
def tune_model(
    hyperparam_grid: List[dict],
    tune_data: StructuredDataset,
    val_size: float,
    random_state: int,
) -> Tuple[LogisticRegression, float]:
    train_data, val_data = split_data(data=tune_data, test_size=val_size, random_state=random_state)
    models = map_task(train_model, concurrency=3)(
        train_args=prepare_train_args(train_data=train_data, hyperparam_grid=hyperparam_grid)
    )
    return get_best_model(models=models, val_data=val_data)


@workflow
def tuning_workflow(
    hyperparam_grid: List[dict],
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> NamedTuple(
    "TuningResults",
    best_model=LogisticRegression,
    best_val_acc=float,
    train_acc=float,
    test_acc=float,
):

    # get and split data
    data = get_data()
    tune_data, test_data = split_data(data=data, test_size=test_size, random_state=random_state)
    
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
        {"C": 1.0},
        {"C": 0.1},
        {"C": 0.01},
        {"C": 0.001},
    ]
    print(f"{tuning_workflow(hyperparam_grid=hyperparam_grid)}")
