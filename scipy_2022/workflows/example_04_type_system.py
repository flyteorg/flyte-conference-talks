"""Reliability: Static Analysis Example"""

from typing import Tuple

import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split

from flytekit import task, workflow

from workflows.example_00_intro import FEATURES, TARGET


@task
def get_data() -> pd.DataFrame:
    """
    ✨ Flyte's rich type system allows for static analysis of the execution graph at
    registration time, raising a compile-time error if the type annotations between
    functions aren't compatible.

    😈 Try changing the output signature of this function to a `dict` and see what
    you get when you register this task.
    """
    penguins = load_penguins()
    return penguins[[TARGET] + FEATURES].dropna()


@task
def split_data(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data, test_size=test_size, random_state=random_state)


@workflow
def get_splits(
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return split_data(data=get_data(), test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    get_splits(test_size=0.2, random_state=123)

# TypeError: Failed to convert return value for var o0 for function get_data with
# error <class 'flytekit.core.type_engine.TypeTransformerFailedError'>: Type of Val
#        species  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
# 0       Adelie            39.1           18.7              181.0       3750.0
# 1       Adelie            39.5           17.4              186.0       3800.0
# 2       Adelie            40.3           18.0              195.0       3250.0
# 4       Adelie            36.7           19.3              193.0       3450.0
# 5       Adelie            39.3           20.6              190.0       3650.0
# ..         ...             ...            ...                ...          ...
# 339  Chinstrap            55.8           19.8              207.0       4000.0
# 340  Chinstrap            43.5           18.1              202.0       3400.0
# 341  Chinstrap            49.6           18.2              193.0       3775.0
# 342  Chinstrap            50.8           19.0              210.0       4100.0
# 343  Chinstrap            50.2           18.7              198.0       3775.0
#
# [342 rows x 5 columns]' is not an instance of <class 'dict'>
