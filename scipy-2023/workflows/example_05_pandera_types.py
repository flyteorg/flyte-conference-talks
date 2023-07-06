"""Reliability: Flytekit Pandera Plugin."""

from typing import NamedTuple

import pandas as pd
import pandera as pa

from pandera.typing import DataFrame, Series
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split

import flytekitplugins.pandera
from flytekit import task, workflow

from workflows.example_00_intro import (
    FEATURES,
    TARGET,
)


CLASSES = ["Adelie", "Gentoo", "Chinstrap"]


class PenguinsSchema(pa.SchemaModel):
    """
    ✅🐼 Pandera schemas are also supported via a type transformer plugin.
    This allows for statistical data validation of dataframes as they flow
    through your Flyte workflows.
    """

    species: Series[str] = pa.Field(isin=CLASSES)
    bill_length_mm: Series[float]
    bill_depth_mm: Series[float]
    flipper_length_mm: Series[float]
    body_mass_g: Series[float]


PenguinDataset = DataFrame[PenguinsSchema]
DataSplits = NamedTuple(
    "DataSplits", train=PenguinDataset, test=PenguinDataset
)


@task
def get_data() -> PenguinDataset:
    penguins = load_penguins()[[TARGET] + FEATURES].dropna()

    # 😈 Uncomment the line below to introduce data corruption
    # penguins = penguins.astype({"bill_length_mm": str})

    return penguins


@task
def split_data(
    data: PenguinDataset, test_size: float, random_state: int
) -> DataSplits:
    return train_test_split(
        data, test_size=test_size, random_state=random_state
    )


@workflow
def get_splits(test_size: float = 0.2, random_state: int = 123) -> DataSplits:
    return split_data(
        data=get_data(), test_size=test_size, random_state=random_state
    )


if __name__ == "__main__":
    train_data, test_data = get_splits()
    print(f"Training data:\n\n{train_data.head(3)}")
    print(f"Test data:\n\n{test_data.head(3)}")
