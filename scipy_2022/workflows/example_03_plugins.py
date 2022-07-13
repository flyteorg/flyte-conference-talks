"""Flyte Plugin Examples."""

from dataclasses import dataclass
from typing import Union

import pandas as pd
import pyspark.pandas as spd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses_json import dataclass_json

from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
from flytekit import kwtypes, task, workflow
from flytekit.extras.sqlite3.task import SQLite3Config, SQLite3Task
from flytekitplugins.spark import Spark

from workflows.example_00_intro import (
    FEATURES,
    TARGET,
)


@dataclass_json
@dataclass
class Hyperparameters:
    in_dim: int
    hidden_dim: int
    out_dim: int
    learning_rate: float


get_data = SQLite3Task(
    name="cookbook.sqlite3.sample",
    query_template="""
    SELECT species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
    FROM penguins
    """,
    output_schema_type=FlyteSchema[
        kwtypes(
            species=str,
            bill_length_mm=float,
            bill_depth_mm=float,
            flipper_length_mm=float,
            body_mass_g=float,
        )
    ],
    task_config=SQLite3Config(uri="https://datasette-seaborn-demo.datasette.io/penguins.db"),
)


def scale(data: Union[pd.DataFrame, spd.DataFrame]):
    return (data - data.mean()) / data.std()


@task
def preprocess_data(data: pd.DataFrame) -> FlyteFile:
    penguins = (
        data[[TARGET] + FEATURES]
        .dropna()
        .sample(frac=1.0, random_state=42)
    )
    penguins[FEATURES] = scale(penguins[FEATURES])
    local_file = "/tmp/penguins.parquet"
    penguins.to_parquet(local_file)
    return FlyteFile(path=local_file)


@task(
    task_config=Spark(
        # this configuration is applied to the spark cluster
        spark_conf={
            "spark.driver.memory": "1000M",
            "spark.executor.instances": "2",
            "spark.driver.cores": "1",
        }
    ),
)
def preprocess_data_pyspark(data: spd.DataFrame) -> FlyteFile:
    penguins = (
        data[[TARGET] + FEATURES]
        .dropna()
        .sample(frac=1.0, random_state=42)
    )
    penguins[FEATURES] = scale(penguins[FEATURES])
    local_file = "/tmp/penguins.parquet"
    penguins.to_parquet(local_file)
    return FlyteFile(path=local_file)


@task
def train_model(data: FlyteFile, n_epochs: int, hyperparameters: Hyperparameters) -> nn.Sequential:
    data = pd.read_parquet(data.path)

    model = nn.Sequential(
        nn.Linear(hyperparameters.in_dim, hyperparameters.hidden_dim),
        nn.ReLU(),
        nn.Linear(hyperparameters.hidden_dim, hyperparameters.out_dim),
        nn.Softmax(dim=1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    features = torch.from_numpy(data[FEATURES].values).float()
    targets = torch.from_numpy(pd.get_dummies(data[TARGET]).values).float()

    for _ in range(n_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(features), targets)
        print(f"loss={loss:.04f}")
        loss.backward()
        opt.step()

    return model


@workflow
def training_workflow(
    n_epochs: int,
    hyperparameters: Hyperparameters,
) -> nn.Sequential:
    data = preprocess_data(data=get_data())
    return train_model(data=data, n_epochs=n_epochs, hyperparameters=hyperparameters)


if __name__ == "__main__":
    hyperparameters = Hyperparameters(in_dim=4, hidden_dim=100, out_dim=3, learning_rate=0.03)
    print(f"{training_workflow(n_epochs=30, hyperparameters=hyperparameters)}")
