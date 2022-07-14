"""Reproducibility: given that random seeds are parameterized in the task."""

from dataclasses import dataclass, asdict

import pandas as pd

from dataclasses_json import dataclass_json
from palmerpenguins import load_penguins
from sklearn.linear_model import SGDClassifier

from flytekit import task, workflow, Resources

from workflows.example_00_intro import get_data, FEATURES, TARGET


@dataclass_json
@dataclass
class Hyperparameters:
    penalty: str = "l2"
    alpha: float = 0.001
    # 🔄 Code-level reproducibility: set random seed within your tasks
    random_state: int = 42


# 🔄 Compute-level reproducibility: encode compute and memory requirements
@task(requests=Resources(cpu="2", mem="750Mi"), limits=Resources(cpu="4", mem="1Gi"))
def get_data() -> pd.DataFrame:
    return load_penguins()[[TARGET] + FEATURES].dropna()


# 💻 Compute/memory requirements can be configured at task-level granularity
@task(requests=Resources(cpu="1", mem="500Mi"), limits=Resources(cpu="2", mem="500Mi"))
def train_model(data: pd.DataFrame, hyperparameters: Hyperparameters) -> SGDClassifier:
    return SGDClassifier(**asdict(hyperparameters)).fit(data[FEATURES], data[TARGET])


@workflow
def training_workflow(hyperparameters: Hyperparameters) -> SGDClassifier:
    return train_model(data=get_data(), hyperparameters=hyperparameters)


if __name__ == "__main__":
    hyperparameters = Hyperparameters(penalty="l1", random_state=12345)
    print(f"{training_workflow(hyperparameters=hyperparameters)}")
