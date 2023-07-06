"""Common testing utilities"""

from typing import Any, Callable, Dict, NamedTuple, Tuple, Type

import pandas as pd
import torch.nn as nn
from flytekit.core.workflow import PythonFunctionWorkflow
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

from workflows import (
    example_00_intro,
    example_01_dynamic,
    example_02_map_task,
    example_03_plugins,
    example_04_type_system,
    example_05_pandera_types,
    example_06_reproducibility,
    example_07_caching,
    example_08_recover_executions,
    example_09_checkpointing,
    example_10_flyte_decks,
    example_11_extend_flyte_decks,
)


class WorkflowCase(NamedTuple):
    workflow: PythonFunctionWorkflow
    inputs: Dict[str, Any]
    expected_output_types: Tuple[Type, ...]


WORKFLOW_CASES = [
    WorkflowCase(
        workflow=example_00_intro.training_workflow,
        inputs={"hyperparameters": example_00_intro.Hyperparameters(C=0.1, max_iter=5000)},
        expected_output_types=(LogisticRegression, float, float),
    ),
    WorkflowCase(
        workflow=example_01_dynamic.tuning_workflow,
        inputs={
            "hyperparam_grid": [
                example_00_intro.Hyperparameters(C=C, max_iter=5000) for C in [1.0, 0.1, 0.01]
            ]
        },
        expected_output_types=example_01_dynamic.TuningResults,
    ),
    WorkflowCase(
        workflow=example_02_map_task.tuning_workflow,
        inputs={
            "hyperparam_grid": [
                example_00_intro.Hyperparameters(C=C, max_iter=5000) for C in [1.0, 0.1, 0.01]
            ]
        },
        expected_output_types=example_01_dynamic.TuningResults,
    ),
    WorkflowCase(
        workflow=example_03_plugins.training_workflow,
        inputs={
            "n_epochs": 2,
            "hyperparameters": example_03_plugins.Hyperparameters(
                in_dim=4, hidden_dim=5, out_dim=3, learning_rate=0.03
            ),
        },
        expected_output_types=nn.Sequential,
    ),
    WorkflowCase(
        workflow=example_04_type_system.get_splits,
        inputs={"test_size": 0.2, "random_state": 123},
        expected_output_types=(pd.DataFrame, pd.DataFrame),
    ),
    WorkflowCase(
        workflow=example_05_pandera_types.get_splits,
        inputs={"test_size": 0.2},
        expected_output_types=example_05_pandera_types.DataSplits,
    ),
    WorkflowCase(
        workflow=example_06_reproducibility.training_workflow,
        inputs={
            "hyperparameters": example_06_reproducibility.Hyperparameters(
                penalty="l2", alpha=0.001, random_state=42,
            ),
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_07_caching.tuning_workflow,
        inputs={
            "hyperparam_grid": [
                example_06_reproducibility.Hyperparameters(alpha=alpha)
                for alpha in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
            ],
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_08_recover_executions.tuning_workflow,
        inputs={
            "alpha_grid": [100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_09_checkpointing.training_workflow,
        inputs={
            "n_epochs": 30,
            "hyperparameters": example_06_reproducibility.Hyperparameters(
                penalty="l1",
                random_state=42,
            ),
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_10_flyte_decks.penguins_data_workflow,
        inputs={},
        expected_output_types=None,
    ),
    WorkflowCase(
        workflow=example_11_extend_flyte_decks.training_workflow,
        inputs={
            "hyperparameters": example_06_reproducibility.Hyperparameters(
                penalty="l1", alpha=0.03, random_state=12345
            )
        },
        expected_output_types=Pipeline,
    ),
]
