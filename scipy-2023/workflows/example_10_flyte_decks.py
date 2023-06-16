"""Auditability: Flyte Decks for Pipeline visibility."""

from typing import Annotated

import pandas as pd
from palmerpenguins import load_penguins

from workflows.example_00_intro import FEATURES, TARGET

from flytekit import task, workflow, Deck, Resources
from flytekitplugins.deck import FrameProfilingRenderer


resources = Resources(mem="1Gi")


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
    return load_penguins()[[TARGET] + FEATURES]


@workflow
def penguins_data_workflow():
    get_data()
    get_data_annotated()


if __name__ == "__main__":
    penguins_data_workflow()
