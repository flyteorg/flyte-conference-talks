"""Flyte Intro: Type and Task Plugin Examples."""

from dataclasses import dataclass
from typing import Annotated

import pandas as pd
import pyspark.sql
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses_json import dataclass_json

from flytekit.types.structured import StructuredDataset
from flytekit import kwtypes, task, workflow, Resources
from flytekit.extras.sqlite3.task import SQLite3Config, SQLite3Task
from flytekitplugins.spark import Spark


from workflows.example_00_intro import FEATURES, TARGET


@dataclass_json
@dataclass
class Hyperparameters:
    in_dim: int
    hidden_dim: int
    out_dim: int
    learning_rate: float


PenquinsDataset = Annotated[
    StructuredDataset,
    kwtypes(
        species=str,
        bill_length_mm=float,
        bill_depth_mm=float,
        flipper_length_mm=float,
        body_mass_g=float,
    ),
]


# 🔌 The first type of plugin in Flyte is the task template plugin.
# These are task objects that can be used like the regular @task
# functions. This SQLite task plugin allows you to perform queries on
# local or remote SQLite databases.
QUERY = "SELECT species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g FROM penguins"
get_data = SQLite3Task(
    name="get_penguins_data",
    query_template=QUERY,
    output_schema_type=PenquinsDataset,
    task_config=SQLite3Config(
        uri="https://datasette-seaborn-demo.datasette.io/penguins.db"
    ),
)

# Other examples of this type of plugin are:
# - SQLAlchemyTask
# - BigQueryTask
# - SnowflakeTask
# - HiveTask
# - AthenaTask
# - FlyteOperator for Airflow


def scale(data: pd.DataFrame):
    """🤝 Helper functions can be invoked in any task."""
    return (data - data.mean()) / data.std()


@task
def preprocess_data(data: PenquinsDataset) -> PenquinsDataset:
    data = data.open(pd.DataFrame).all()
    penguins = (
        data[[TARGET] + FEATURES].dropna().sample(frac=1.0, random_state=42)
    )
    penguins[FEATURES] = scale(penguins[FEATURES])
    return PenquinsDataset(penguins)


@task(
    task_config=Spark(
        spark_conf={
            "spark.driver.memory": "1000M",
            "spark.executor.instances": "2",
            "spark.driver.cores": "1",
        }
    ),
)
def preprocess_data_pyspark(
    data: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    🔌 The second type of task plugin is the task configuration plugin. For
    example, the Spark plugin allows you to declaratively specify the compute
    and memory requirements of an ephemeral Spark cluster, which is set up and
    torn down automatically by Flyte.

    Other examples of this type of plugin are:
    - Ray operator
    - Sagemaker operator
    - MPI operator
    - Kubeflow Pytorch operator
    - Kubeflow Tensorflow operator
    """
    ...  # pyspark code


RESOURCES = Resources(cpu="2", mem="1Gi")


@task(requests=RESOURCES, limits=RESOURCES)
def train_model(
    data: PenquinsDataset, n_epochs: int, hyperparameters: Hyperparameters
) -> nn.Sequential:
    """
    🔌 The third kind of plugin is the type transformer plugin, which enables you
    to support types that Flyte doesn't ship with out-of-the-box. Pytorch modules,
    like `nn.Sequential`, are supported in the flytekit.extras module, but virtually
    any type in Python can be understood by Flyte.

    Other examples of this type of plugin are:
    - Modin type
    - Pandera type
    - ONNX type
    """
    # extract features and targets
    data = data.open(pd.DataFrame).all()
    features = torch.from_numpy(data[FEATURES].values).float()
    targets = torch.from_numpy(pd.get_dummies(data[TARGET]).values).float()

    # create model
    model = nn.Sequential(
        nn.Linear(hyperparameters.in_dim, hyperparameters.hidden_dim),
        nn.ReLU(),
        nn.Linear(hyperparameters.hidden_dim, hyperparameters.out_dim),
        nn.Softmax(dim=1),
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=hyperparameters.learning_rate
    )

    # train for n_epochs
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
    return train_model(
        data=data, n_epochs=n_epochs, hyperparameters=hyperparameters
    )


if __name__ == "__main__":
    hyperparameters = Hyperparameters(
        in_dim=4, hidden_dim=100, out_dim=3, learning_rate=0.03
    )
    print(f"{training_workflow(n_epochs=30, hyperparameters=hyperparameters)}")
