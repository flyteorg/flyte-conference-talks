"""Reproducibility: given that random seeds are parameterized in the task."""

from dataclasses import asdict, dataclass
from typing import List

import pandas as pd
from dataclasses_json import dataclass_json
from flytekit import ImageSpec, Resources, task, workflow
from sklearn.linear_model import SGDClassifier
from workflows.example_00_intro import FEATURES, TARGET, get_data

custom_image = ImageSpec(registry="samhitaalla", packages=["palmerpenguins"])

if custom_image.is_container():
    from palmerpenguins import load_penguins


@dataclass_json
@dataclass
class Hyperparameters:
    penalty: str = "l2"
    alpha: float = 0.001
    # 🔄 Code-level reproducibility: set random seed within your tasks
    random_state: int = 42


# 🔄 Compute-level reproducibility: encode compute and memory requirements
@task(
    requests=Resources(cpu="2", mem="1Gi"),
    limits=Resources(cpu="2", mem="1Gi"),
    container_image=custom_image,
)
def get_data() -> pd.DataFrame:
    return load_penguins()[[TARGET] + FEATURES].dropna()


# 💻 Compute/memory requirements can be configured at task-level granularity
@task(requests=Resources(cpu="1", mem="500Mi"), limits=Resources(cpu="1", mem="750Mi"))
def train_model(data: pd.DataFrame, hyperparameters: Hyperparameters) -> SGDClassifier:
    return SGDClassifier(**asdict(hyperparameters)).fit(data[FEATURES], data[TARGET])


@task
def model_weights(model: SGDClassifier) -> List[List[float]]:
    return [[float(x) for x in coef] for coef in model.coef_.tolist()]


@workflow
def training_workflow(hyperparameters: Hyperparameters) -> SGDClassifier:
    model = train_model(data=get_data(), hyperparameters=hyperparameters)
    model_weights(model=model)
    return model


if __name__ == "__main__":
    hyperparameters = Hyperparameters(penalty="l1", random_state=12345)
    print(f"{training_workflow(hyperparameters=hyperparameters)}")
