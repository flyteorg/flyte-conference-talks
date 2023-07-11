"""Flyte Intro: Tasks and Workflows.

These examples will use the penguins dataset:
https://allisonhorst.github.io/palmerpenguins/

Using the pypi package:
https://pypi.org/project/palmerpenguins/
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import pandas as pd
from dataclasses_json import dataclass_json
from palmerpenguins import load_penguins
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from flytekit import task, workflow, LaunchPlan, CronSchedule

try:
    from workflows import logger
except:
    pass


TARGET = "species"
FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

@dataclass_json
@dataclass
class Hyperparameters:
    C: float
    max_iter: Optional[int] = 2500


@task
def get_data() -> pd.DataFrame:
    return load_penguins()[[TARGET] + FEATURES].dropna()


@task
def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )


@task
def train_model(
    data: pd.DataFrame,
    hyperparameters: Hyperparameters,
) -> LogisticRegression:
    return LogisticRegression(**asdict(hyperparameters)).fit(
        data[FEATURES], data[TARGET]
    )


@task
def evaluate(model: LogisticRegression, data: pd.DataFrame) -> float:
    return float(accuracy_score(data[TARGET], model.predict(data[FEATURES])))


@workflow
def training_workflow(
    hyperparameters: Hyperparameters,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[LogisticRegression, float, float]:
    # get and split data
    data = get_data()
    train_data, test_data = split_data(
        data=data, test_size=test_size, random_state=random_state
    )

    # train model on the training set
    model = train_model(data=train_data, hyperparameters=hyperparameters)

    # evaluate the model
    train_acc = evaluate(model=model, data=train_data)
    test_acc = evaluate(model=model, data=test_data)

    # return model with results
    return model, train_acc, test_acc


training_launchplan = LaunchPlan.get_or_create(
    training_workflow,

    name="scheduled_training_workflow",

    # run every 2 minutes
    schedule=CronSchedule(schedule="*/2 * * * *"),

    # use default inputs
    default_inputs={"hyperparameters": Hyperparameters(C=0.1, max_iter=1000)},
)


if __name__ == "__main__":
    # You can run workflows locally, it's just Python 🐍!
    print(f"{training_workflow(hyperparameters=Hyperparameters(C=0.1, max_iter=5000))}")
