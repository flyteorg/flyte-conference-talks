
import datetime
import random
from operator import add

import flytekit
from flytekit import Resources, task, workflow
from flytekitplugins.spark import Spark


@task(
    task_config=Spark(
        # this configuration is applied to the spark cluster
        spark_conf={
            "spark.driver.memory": "1000M",
            "spark.executor.memory": "1000M",
            "spark.executor.cores": "1",
            "spark.executor.instances": "2",
            "spark.driver.cores": "1",
        }
    ),
    limits=Resources(mem="2000M"),
    cache_version="1",
)
def hello_spark(partitions: int) -> float:
    print("Starting Spark with Partitions: {}".format(partitions))

    n = 1000 * partitions
    sess = flytekit.current_context().spark_session
    count = (
        sess.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
    )
    pi_val = 4.0 * count / n
    print("Pi val is: {}".format(pi_val))
    return pi_val


def f(_):
    x = random.random() * 2 - 1
    y = random.random() * 2 - 1
    return 1 if x**2 + y**2 <= 1 else 0


@task(cache_version="1")
def print_every_time(value_to_print: float, date_triggered: datetime.datetime) -> int:
    print("My printed value: {} @ {}".format(value_to_print, date_triggered))
    return 1


@workflow
def my_spark(triggered_date: datetime.datetime) -> float:
    pi = hello_spark(partitions=50)
    print_every_time(value_to_print=pi, date_triggered=triggered_date)
    return pi


if __name__ == "__main__":
    print(
        f"Running my_spark(triggered_date=datetime.datetime.now()){my_spark(triggered_date=datetime.datetime.now())}"
    )
