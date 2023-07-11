"""Auditability: Flyte Decks for Pipeline visibility."""

import pandas as pd
from dataclasses import asdict
from palmerpenguins import load_penguins

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import mlflow
import whylogs as why
from flytekit import task, workflow, Deck, Resources
from flytekitplugins.deck import FrameProfilingRenderer
from flytekitplugins.mlflow import mlflow_autolog
from flytekitplugins.whylogs.renderer import WhylogsConstraintsRenderer
from sklearn.linear_model import LogisticRegression
from whylogs.core import DatasetProfileView
from whylogs.core.constraints import ConstraintsBuilder
from whylogs.core.constraints.factories import (
    greater_than_number,
    mean_between_range,
    null_percentage_below_number,
    smaller_than_number,
)

from workflows.example_00_intro import FEATURES, TARGET, Hyperparameters


resources = Resources(mem="4Gi")


@task(requests=resources, limits=resources, disable_deck=False)
def get_data() -> pd.DataFrame:
    """
    🃏 Flyte Decks allow you to render html in the Flyte console so you can
    visualize and document metadata associated with a task.
    """
    penguins = load_penguins()[[TARGET] + FEATURES]
    Deck("data_profile", FrameProfilingRenderer("penguins").to_html(penguins))
    return penguins


@task(requests=resources, limits=resources, disable_deck=False)
def get_data_annotated() -> Annotated[
    pd.DataFrame, FrameProfilingRenderer("penguins")
]:
    """
    🃏 Flyte Decks can also be rendered at the output interface of your tasks.
    """
    return load_penguins()[[TARGET] + FEATURES].dropna()


@task(requests=resources, limits=resources, disable_deck=False)
def get_profile_view(data: pd.DataFrame) -> DatasetProfileView:
    result = why.log(data)
    return result.view()


@task(requests=resources, limits=resources, disable_deck=False)
def get_constraints_report(profile_view: DatasetProfileView) -> bool:
    builder = ConstraintsBuilder(dataset_profile_view=profile_view)
    builder.add_constraint(greater_than_number("bill_length_mm", 0))
    builder.add_constraint(mean_between_range("bill_depth_mm", 0, 100))
    builder.add_constraint(null_percentage_below_number("flipper_length_mm", 0.1))
    builder.add_constraint(smaller_than_number("body_mass_g", 10_000))
    constraints = builder.build()

    renderer = WhylogsConstraintsRenderer()
    Deck("constraints", renderer.to_html(constraints=constraints))
    return constraints.validate()


@task(disable_deck=False)
@mlflow_autolog(
    framework=mlflow.sklearn,
    experiment_name="penguins_experiment",
)
def train_model(
    data: pd.DataFrame,
    hyperparameters: Hyperparameters,
) -> LogisticRegression:
    return LogisticRegression(**asdict(hyperparameters)).fit(
        data[FEATURES], data[TARGET]
    )


@workflow
def penguins_data_workflow(
    hyperparameters: Hyperparameters = Hyperparameters(C=0.01)
) -> LogisticRegression:
    data = get_data()
    annotated_data = get_data_annotated()

    profile_view = get_profile_view(data=data)
    get_constraints_report(profile_view=profile_view)

    return train_model(data=annotated_data, hyperparameters=hyperparameters)


if __name__ == "__main__":
    penguins_data_workflow()
