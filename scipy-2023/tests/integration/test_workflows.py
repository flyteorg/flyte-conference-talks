"""Script to run all the workflows on a remote Flyte cluster.

NOTE: This script assumes that:
1. You have the appropriate configuration to run executions on the remote cluster.
2. The workflows are registered in the cluster.
"""

import logging
import os
import time
from datetime import timedelta

from flytekit.remote import FlyteRemote
from flytekit.configuration import Config
from pathlib import Path

import pytest

from tests.common import WorkflowCase, WORKFLOW_CASES


logger = logging.getLogger(__name__)


SUCCEED_STATUS = 4

CONFIG_PATH = os.environ.get("FLYTECTL_CONFIG", None)
if CONFIG_PATH is None:
    config = Config.for_sandbox()
else:
    config = Config.auto(CONFIG_PATH)

if int(os.environ.get("CI", 0)):
    workflow_cases = WORKFLOW_CASES[:1]
    poll_interval = timedelta(seconds=90)
else:
    workflow_cases = WORKFLOW_CASES
    poll_interval = None


remote = FlyteRemote(
    config=config,
    default_project="flytesnacks",
    default_domain="development",
)


@pytest.mark.parametrize("wf_case", workflow_cases)
def test_workflow_remote(wf_case: WorkflowCase):
    for _ in range(120):
        # bypass issue where multiple remote objects are authenticating at the
        # same time.
        try:
            flyte_wf = remote.fetch_workflow(name=wf_case.workflow.name)
            break
        except Exception:
            time.sleep(5)

    execution = remote.execute(flyte_wf, inputs=wf_case.inputs)
    url = remote.generate_console_url(execution)
    logger.info(f"Running workflow {wf_case.workflow.name} at: {url}")
    execution = remote.wait(execution, poll_interval=poll_interval)
    assert execution.closure.phase == SUCCEED_STATUS
