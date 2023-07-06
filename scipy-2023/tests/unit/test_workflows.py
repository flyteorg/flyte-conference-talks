"""Unit tests for Flyte workflows."""

import pytest

from unittest.mock import patch

from tests.common import WorkflowCase, WORKFLOW_CASES


def _is_namedtuple(type_):
    try:
        return issubclass(type_, tuple) and hasattr(type_, "_fields")
    except TypeError:
        return False



@pytest.mark.parametrize("wf_case", WORKFLOW_CASES)
@patch("workflows.example_08_recover_executions.FAILURE_RATE", 0.0)
@patch("workflows.example_09_checkpointing.FAILURE_RATE", 0.0)
def test_workflow(wf_case: WorkflowCase, *_):
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
