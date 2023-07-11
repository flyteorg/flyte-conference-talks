"""Example of using container tasks."""

import pandas as pd

from flytekit import task, workflow, kwtypes, ContainerTask, TaskMetadata
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset


PENGUINS_DATASET_URL = "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/master/palmerpenguins/data/penguins.csv"


get_dataset = ContainerTask(
    name="get_dataset",
    image="ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-wget-latest",
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    inputs=kwtypes(url=str),
    outputs=kwtypes(dataset=FlyteFile),
    command=[
        "./scripts/download_dataset.sh",
        "{{.inputs.url}}",
        "/var/outputs/dataset"
    ],
    metadata=TaskMetadata(retries=5)
)


@task
def convert_to_structured_dataset(dataset: FlyteFile) -> StructuredDataset:
    with open(dataset, "r") as f:
        df = pd.read_csv(f)
    return StructuredDataset(df)


@workflow
def get_data_wf(url: str = PENGUINS_DATASET_URL) -> StructuredDataset:
    dataset = get_dataset(url=url)
    return convert_to_structured_dataset(dataset=dataset)
