"""Recoverability: Checkpoints to recover execution progress within tasks."""

from io import BytesIO
from dataclasses import asdict
from random import random

import joblib
import pandas as pd

from sklearn.linear_model import SGDClassifier

from flytekit import task, workflow, current_context
from flytekit.exceptions.user import FlyteRecoverableException

from workflows.example_05_pandera_types import CLASSES
from workflows.example_06_reproducibility import (
    get_data,
    Hyperparameters,
    FEATURES,
    TARGET,
)


@task(retries=10)
def train_model(
    data: pd.DataFrame,
    n_epochs: int,
    hyperparameters: Hyperparameters,
) -> SGDClassifier:
    """
    Caching and workflow recovery is great, but what if your training
    task fails before completing?

    ✨ Use intra-task checkpoints to resume training within a task so that
    you don't have to start from scratch.
    """

    # try to get previous checkpoint, if it exists
    try:
        checkpoint = current_context().checkpoint
        prev_checkpoint = checkpoint.read()
    except (NotImplementedError, ValueError):
        checkpoint, prev_checkpoint = None, False

    # assume that checkpoint consists of a counter of the latest epoch and model
    if prev_checkpoint:
        start_epoch, model = joblib.load(BytesIO(prev_checkpoint))
    else:
        start_epoch, model = 0, SGDClassifier(**asdict(hyperparameters))

    for epoch in range(start_epoch, n_epochs):
        print(f"epoch: {epoch}")

        # simulate system-level error: per epoch, introduce
        # a chance of failure 5% of the time
        if random() < 0.05:
            raise FlyteRecoverableException(
                f"🔥 Something went wrong at epoch {epoch}! 🔥"
            )

        model.partial_fit(data[FEATURES], data[TARGET], classes=CLASSES)

        # checkpoint at every epoch
        if checkpoint:
            model_io = BytesIO()
            joblib.dump((epoch, model), model_io)
            model_io.seek(0)
            checkpoint.write(model_io.read())

    return model


@workflow
def training_workflow(
    n_epochs: int, hyperparameters: Hyperparameters
) -> SGDClassifier:
    data = get_data()
    model = train_model(
        data=data, n_epochs=n_epochs, hyperparameters=hyperparameters
    )
    return model


if __name__ == "__main__":
    hyperparameters = Hyperparameters(penalty="l1", random_state=12345)
    print(
        f"{training_workflow(n_epochs=100, hyperparameters=hyperparameters)}"
    )
