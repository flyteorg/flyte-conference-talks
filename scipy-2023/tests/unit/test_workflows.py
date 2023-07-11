"""Unit tests for Flyte workflows."""

import pytest
import subprocess
import tempfile
from unittest.mock import patch

import pandas as pd
from flytekit.types.file import FlyteFile
from flytekit.testing import task_mock

from tests.common import WorkflowCase, WORKFLOW_CASES


@pytest.fixture
def clear_cache():
    subprocess.run(["pyflyte", "local-cache", "clear"])


def _is_namedtuple(type_):
    try:
        return issubclass(type_, tuple) and hasattr(type_, "_fields")
    except TypeError:
        return False
    

def container_task_wf(wf_case: WorkflowCase):
    from workflows.example_container_tasks import get_dataset

    with task_mock(get_dataset) as mock:
        with tempfile.NamedTemporaryFile() as f:
            test_dataset = pd.DataFrame({"a": [1, 2, 3]})
            test_dataset.to_csv(f)
            mock.return_value = FlyteFile(f.name)
            return wf_case.workflow(**wf_case.inputs)

@pytest.mark.parametrize("wf_case", WORKFLOW_CASES)
@patch("workflows.example_recover_executions.FAILURE_RATE", 0.0)
@patch("workflows.example_checkpointing.FAILURE_RATE", 0.0)
def test_workflow(wf_case: WorkflowCase, clear_cache, *_):
    if wf_case.workflow.name == "workflows.example_container_tasks.get_data_wf":
        output = container_task_wf(wf_case)
    else:
        output = wf_case.workflow(**wf_case.inputs)

    if wf_case.expected_output_types is None:
        return
    elif _is_namedtuple(wf_case.expected_output_types):
        # handle named tuple case
        assert wf_case.expected_output_types(*output)
    elif isinstance(wf_case.expected_output_types, tuple):
        for output, expected_type in zip(output, wf_case.expected_output_types):
            if not expected_type:
                return
            assert isinstance(output, expected_type)
    else:
        assert isinstance(output, wf_case.expected_output_types)
