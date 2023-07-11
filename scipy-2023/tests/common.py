"""Common testing utilities"""

from typing import Any, Dict, NamedTuple, Tuple, Type, Union

import pandas as pd
import torch.nn as nn
from flytekit.core.workflow import PythonFunctionWorkflow
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

from workflows import (
    example_caching,
    example_checkpointing,
    example_container_tasks,
    example_dynamic,
    example_flyte_decks,
    example_flyte_decks_extend,
    example_intro,
    example_map_task,
    example_notebook_tasks,
    example_pandera_types,
    example_plugins,
    example_recover_executions,
    example_reproducibility,
    example_type_system,
)


class WorkflowCase(NamedTuple):
    workflow: PythonFunctionWorkflow
    inputs: Dict[str, Any]
    expected_output_types: Union[Type, Tuple[Type, ...]]


WORKFLOW_CASES = [
    WorkflowCase(
        workflow=example_intro.training_workflow,
        inputs={"hyperparameters": example_intro.Hyperparameters(C=0.1, max_iter=5000)},
        expected_output_types=(LogisticRegression, float, float),
    ),
    WorkflowCase(
        workflow=example_notebook_tasks.data_analysis_wf,
        inputs={},
        expected_output_types=None,
    ),
    WorkflowCase(
        workflow=example_container_tasks.get_data_wf,
        inputs={"url": example_container_tasks.PENGUINS_DATASET_URL},
        expected_output_types=StructuredDataset,
    ),
    WorkflowCase(
        workflow=example_dynamic.tuning_workflow,
        inputs={
            "hyperparam_grid": [
                example_intro.Hyperparameters(C=C, max_iter=5000) for C in [1.0, 0.1, 0.01]
            ]
        },
        expected_output_types=example_dynamic.TuningResults,
    ),
    WorkflowCase(
        workflow=example_map_task.tuning_workflow,
        inputs={
            "hyperparam_grid": [
                example_intro.Hyperparameters(C=C, max_iter=5000) for C in [1.0, 0.1, 0.01]
            ]
        },
        expected_output_types=example_dynamic.TuningResults,
    ),
    WorkflowCase(
        workflow=example_plugins.training_workflow,
        inputs={
            "n_epochs": 2,
            "hyperparameters": example_plugins.Hyperparameters(
                in_dim=4, hidden_dim=5, out_dim=3, learning_rate=0.03
            ),
        },
        expected_output_types=nn.Sequential,
    ),
    WorkflowCase(
        workflow=example_type_system.get_splits,
        inputs={"test_size": 0.2, "random_state": 123},
        expected_output_types=(pd.DataFrame, pd.DataFrame),
    ),
    WorkflowCase(
        workflow=example_pandera_types.get_splits,
        inputs={"test_size": 0.2},
        expected_output_types=example_pandera_types.DataSplits,
    ),
    WorkflowCase(
        workflow=example_reproducibility.training_workflow,
        inputs={
            "hyperparameters": example_reproducibility.Hyperparameters(
                penalty="l2", alpha=0.001, random_state=42,
            ),
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_caching.tuning_workflow,
        inputs={
            "hyperparam_grid": [
                example_reproducibility.Hyperparameters(alpha=alpha)
                for alpha in [0.1, 0.01, 0.001]
            ],
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_recover_executions.tuning_workflow,
        inputs={
            "alpha_grid": [0.1, 0.01, 0.001],
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_checkpointing.training_workflow,
        inputs={
            "n_epochs": 30,
            "hyperparameters": example_reproducibility.Hyperparameters(
                penalty="l1",
                random_state=42,
            ),
        },
        expected_output_types=SGDClassifier,
    ),
    WorkflowCase(
        workflow=example_flyte_decks.penguins_data_workflow,
        inputs={},
        expected_output_types=LogisticRegression,
    ),
    WorkflowCase(
        workflow=example_flyte_decks_extend.training_workflow,
        inputs={
            "hyperparameters": example_reproducibility.Hyperparameters(
                penalty="l1", alpha=0.03, random_state=12345
            )
        },
        expected_output_types=Pipeline,
    ),
]
