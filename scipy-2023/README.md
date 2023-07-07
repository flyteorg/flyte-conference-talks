# Flyte Tutorial: Scipy 2023

This directory contains the workshop materials for [Scipy 2023](https://cfp.scipy.org/2023/talk/YHEYVY/).

## Production-grade Data Science Orchestration with Flyte

This tutorial will focus on four challenges in the context of data science in
production:

- Scalability
- Data Quality
- Reproducibility
- Recoverability
- Auditability

Using Flyte, a data- and machine-learning open source orchestrator, we’ll see
how to address these challenges and abstract them out to give you a broader
understanding of how to surmount them.

First we'll define and describe what these four challenges mean in the context
of ML model development. Then I’ll dive into the ways in which Flyte provides
solutions to them, taking you through the reasoning behind Flyte’s data-centric
and ML-aware design.

You'll learn how Flyte distributes and scales computation, enforces static and
runtime type safety, leverages Docker to provide strong reproducibility
guarantees, implements caching and checkpointing to recover from failed model
training runs, and ships with built-in data lineage tracking for full data
pipeline auditability.

## Outline

| 🔤 Introduction to Flyte [45 minutes] | |
| --- | --- |
| **Environment Setup** | Setting up your virtual development environment |
| **Tasks, Workflows, and Launch Plans** | The building blocks for expressing execution graphs |
| **Scheduling Launch Plans** | Run your workflows on a schedule and get notified about their status |
| **`pyflyte run`** | Run tasks and workflows locally or on a Flyte cluster  |
| **Flyte Console** | A tour of the Flyte console to view workflow progress and status  |
| **`FlyteRemote`** | Programmatically run tasks and workflows  |

> ⏱️ 15 minute break

| 🔀 Flyte Programming Model [45 minutes] | |
| --- | --- |
| **Development Lifecycle Overview** | How to progress from development to production |
| **Tasks as Containerized Functions** | A core building block for statelessness and reproducibility |
| **Container Tasks** | Incorporate tasks written in any language into your workflows |
| **How Data Flows in Flyte** | How the type system helps to abstract away passing data between tasks |
| **Primitive Types vs. Offloaded Types** | How Flyte handles different types |
| **Data and Machine Learning Types** | Type-handling for data- and ML-specific objects |
| **Lifecycle of a Workflow** | How workflows are executed by a Flyte backend |

> ⏱️ 15 minute break

| 🚀 Productionizing Data Science Workloads [45 minutes] | |
| --- | --- |
| **Parallelism** | Use dynamic workflows and map tasks to parallelize your tasks |
| **Resource Allocation** | Specify heterogenous resources requirements at the task-level |
| **Scaling** | Use the SQL, Spark, and Pytorch Elastic plugins to scale your workloads |
| **Production Notebooks** | Use `NotebookTask`s to leverate jupyter notebooks in production workflows |
| **ImageSpec** | Abstracting the containerization step with ImageSpec |
| **Recovering from Failure** | Build robust pipelines wiht retries, caching, failure recovery, and checkpointing |
| **Auditing Workflows** | Gain visibility into your tasks with Flyte Decks |

> ⏱️ 15 minute break

| 🔎 Testing, CI/CD, Extending Flyte [45 minutes] | |
| --- | --- |
| **Writing Unit Tests** | Test Flyte tasks and workflows in isolation |
| **Writing Integration Tests** | Test Flyte workflows on a local cluster |
| **Using Github Actions** | Use github actions to automate testing |
| **Extending Flyte** | Extend Flyte with decorators, type plugins, flyte deck extensions, and task plugins |

> 🗣️ 15 minute Q&A

## Prerequisites

> ⚠️ **Note:** Windows users need to have [WSL installed](https://docs.microsoft.com/en-us/windows/wsl/install-win10) in order to run this workshop.

- Install [Python >= 3.8](https://www.python.org/downloads/)
- Install [Docker Desktop](https://docs.docker.com/get-docker/) and make sure the Docker daemon is running.
- Install `flytectl`:
   ```bash
   # Homebrew (MacOS)
   brew install flyteorg/homebrew-tap/flytectl

   # Or Curl
   curl -sL https://ctl.flyte.org/install | sudo bash -s -- -b /usr/local/bin
   ```

## Setup

Create a fork of this repo by going by going to the
[repo link](https://github.com/flyteorg/flyte-conference-talks) and clicking
on the **Fork** button on the top right of the page.

Clone this repo and go to the workshop directory, replacing `<username>` with
your username:

```bash
git clone https://github.com/<username>/flyte-conference-talks
cd flyte-conference-talks/scipy-2023
```

Create a virtual environment:

```bash
python -m venv ~/venvs/scipy-2023
source ~/venvs/scipy-2023/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt jupyter ipdb
```

Test the virtual environment with:

```bash
pyflyte run \
    workflows/example_00_intro.py training_workflow \
    --hyperparameters '{"C": 0.01}'
```

### Start a local Flyte sandbox:

> ℹ **Note**: Before you start the local cluster, make sure that you allocate a minimum of 4 CPUs and 3 GB of memory in your Docker daemon. If you’re using the **Docker Desktop** application, you can do this easily by going to:
>
> `Settings > Resources > Advanced`
>
> Then set the CPUs and Memory sliders to the appropriate levels.


```bash
flytectl demo start
export FLYTECTL_CONFIG=~/.flyte/config-sandbox.yaml

# update task resources
flytectl update task-resource-attribute --attrFile cra.yaml
```

> ℹ **Note**: Go to the [Troubleshooting](#troubleshooting) section if you're
> having trouble getting the sandbox to start.

Test the Flyte sandbox with:

```bash
export IMAGE=ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-ccfab5b3da86323f07a643ab576d0ad3ed37e3ea

pyflyte run --remote \
    --image $IMAGE \
    workflows/example_00_intro.py training_workflow \
    --hyperparameters '{"C": 0.01}'
```


## Tests

Install dev dependencies:

```bash
pip install pytest pytest-xdist
```

### Unit tests:

```bash
pytest tests/unit
```

### End-to-end tests:

First register all the workflows:

```bash
pyflyte register --image $IMAGE workflows
```

Then run the end-to-end pytest suite:

```bash
pytest tests/end_to_end -n auto
```

## Troubleshooting

### `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.`

You may need to allow the default Docker socket to be used by third-party clients.
Enable this by going to the **Docker Desktop** application and navigating to:

`Settings > Advanced`

Then, click on the checkbox next to **Allow the default Docker socket to be used**,
then **Apply & restart**.
