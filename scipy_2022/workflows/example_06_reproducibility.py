"""Reproducibility, given that random seeds are parameterized in the task."""

from dataclasses import dataclass, asdict

import pandas as pd

from dataclasses_json import dataclass_json
from palmerpenguins import load_penguins
from sklearn.linear_model import SGDClassifier

from flytekit import task, workflow, Resources

from workflows.example_00_intro import (
    get_data,
    FEATURES,
    TARGET,
)


@dataclass_json
@dataclass
class Hyperparameters:
    penalty: str = "l2"
    alpha: float = 0.001
    random_state: int = 42


@task(requests=Resources(cpu="2", mem="750Mi"), limits=Resources(cpu="4", mem="1Gi"))
def get_data() -> pd.DataFrame:
    penguins = load_penguins()
    return penguins[[TARGET] + FEATURES].dropna()


@task(requests=Resources(cpu="1", mem="500Mi"), limits=Resources(cpu="2", mem="500Mi"))
def train_model(data: pd.DataFrame, hyperparameters: Hyperparameters) -> SGDClassifier:
    model = SGDClassifier(**asdict(hyperparameters))
    return model.fit(data[FEATURES], data[TARGET])


@workflow
def training_workflow(hyperparameters: Hyperparameters) -> SGDClassifier:
    data = get_data()    
    model = train_model(data=data, hyperparameters=hyperparameters)
    return model


if __name__ == "__main__":
    hyperparameters = Hyperparameters(penalty="l1", random_state=12345)
    print(f"{training_workflow(hyperparameters=hyperparameters)}")
