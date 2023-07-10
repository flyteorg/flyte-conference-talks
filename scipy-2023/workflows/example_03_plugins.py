"""Flyte Intro: Type and Task Plugin Examples."""

import os
from dataclasses import dataclass

import pandas as pd
import pyspark.sql
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses_json import dataclass_json
from torch import distributed as dist

from flytekit.types.structured import StructuredDataset
from flytekit import kwtypes, task, workflow, Resources
from flytekit.extras.sqlite3.task import SQLite3Config, SQLite3Task
from flytekitplugins.spark import Spark
from flytekitplugins.kfpytorch import Elastic

from workflows.example_00_intro import FEATURES, TARGET

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated


@dataclass_json
@dataclass
class Hyperparameters:
    in_dim: int
    hidden_dim: int
    out_dim: int
    learning_rate: float
    backend: str = dist.Backend.GLOO


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
    container_image="ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-ccfab5b3da86323f07a643ab576d0ad3ed37e3ea",
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


@task(requests=Resources(cpu="2", mem="4Gi"))
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


if int(os.environ.get("USE_ELASTIC", 0)):
    training_config = Elastic(nnodes=1)
    training_resources = Resources(cpu="2", mem="4Gi", gpu="1")
else:
    training_config = None
    training_resources = Resources(cpu="2", mem="4Gi")


@task(
    task_config=training_config,
    requests=training_resources,
)
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if dist.is_available() and world_size > 1:
        print("Using distributed PyTorch with {} backend".format(hyperparameters.backend))
        dist.init_process_group(backend=hyperparameters.backend)

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
    ).to(device)

    if dist.is_available() and dist.is_initialized():
        Distributor = (
            nn.parallel.DistributedDataParallel
            if use_cuda
            else nn.parallel.DistributedDataParallelCPU
        )
        model = Distributor(model)

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
