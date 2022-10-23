"""
Ray task allows you to run a Ray job on an existing Ray cluster or create a Ray cluster using the Ray operator.
"""
import typing

import ray
from flytekit import Resources, task, workflow
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig


@ray.remote
def f(x):
    return x * x


ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=2)],
    runtime_env={"pip": ["numpy", "pandas"]},  # or runtime_env="./requirements.txt"
)


@task(task_config=ray_config, limits=Resources(mem="2000Mi", cpu="1"))
def ray_task(n: int) -> typing.List[int]:
    """
    The task will be called in the Ray head node.
    f.remote(i) will be executed asynchronously on separate Ray workers.
    """
    futures = [f.remote(i) for i in range(n)]
    return ray.get(futures)


@workflow
def ray_workflow(n: int) -> typing.List[int]:
    return ray_task(n=n)


if __name__ == "__main__":
    print(ray_workflow(n=10))
