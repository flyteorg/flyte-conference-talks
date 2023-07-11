"""An example of how to use Jupyter Notebooks in your workflows."""

from pathlib import Path
from typing import Tuple

from flytekit import workflow, kwtypes
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset
from flytekitplugins.papermill import NotebookTask


from workflows.example_00_intro import get_data


data_analysis_task = NotebookTask(
    name="workflows.data_analysis_task",
    notebook_path=str(
        Path(__file__).parent.parent.absolute()
        / "notebooks"
        / "data_analysis_task.ipynb"
    ),
    render_deck=True,
    disable_deck=False,
    inputs=kwtypes(structured_dataset=StructuredDataset),
    outputs=kwtypes(success=bool),
)

@workflow
def data_analysis_wf():
    data = get_data()
    data_analysis_task(structured_dataset=data)
