<html>
    <p align="center">
        <img src="https://github.com/flyteorg/flyte/blob/master/rsts/images/flyte_circle_gradient_1_4x4.png" alt="Flyte Logo" width="100">
    </p>
    <h1 align="center">
        Kubecon 2022 Demo
    </h1>
</html>

## Setup

- Install requirements: `pip install -r requirements.txt`
- [Install Flytectl](https://docs.flyte.org/projects/flytectl/en/latest/#installation)
- Start a Flyte demo cluster: `flytectl sandbox start --source .`
- Build and push a Docker image comprising code and all the requirements.
  - [GitHub registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
  - [Dockerhub](https://docs.docker.com/docker-hub/)

## Task & Workflow

- Test locally:
    ```
    python workflows/example_00_intro.py
    ```
- Test on demo cluster:
    ```
    pyflyte run --remote --image ghcr.io/samhita-alla/kubecon-demo:0.0.2 workflows/example_00_intro.py training_workflow
    ```

## Ray

- Test locally:
    ```
    python workflows/example_01_ray.py
    ```
- [Set up the backend plugin](https://docs.flyte.org/en/latest/deployment/plugin_setup/k8s/index.html)
- Test on demo cluster:
    ```
    pyflyte run --remote --image ghcr.io/samhita-alla/kubecon-demo:0.0.2 workflows/example_01_ray.py ray_workflow --n 3
    ```

## Spark

- Test locally:
    ```
    python workflows/example_02_spark.py
    ```
- [Set up the backend plugin](https://docs.flyte.org/en/latest/deployment/plugin_setup/k8s/index.html)
- Test on demo cluster:
    ```
    pyflyte run --remote --image ghcr.io/samhita-alla/kubecon-spark:0.0.3 workflows/example_02_spark.py my_spark --triggered_date 2022-10-20
    ```

## Map Task

- Test locally:
    ```
    python -m workflows.example_03_map_task
    ```
- Test on demo cluster:
    ```
    pyflyte run --remote --image ghcr.io/samhita-alla/kubecon-demo:0.0.2 workflows/example_03_map_task.py tuning_workflow
    ```

## Resources

- [Flyte cheat sheet](https://raw.githubusercontent.com/flyteorg/static-resources/main/cheatsheets/flyte_cheat_sheet.pdf)
- [Documentation](https://docs.flyte.org/)