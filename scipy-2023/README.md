# Flyte Tutorial: Scipy 2023

This directory contains the workshop materials for [Scipy 2023](https://cfp.scipy.org/2023/talk/YHEYVY/).

## Production-grade Data Science Orchestration with Flyte

This workshop will focus on five facets of production-grade data science:

- ⛰️ Scalability
- ✅ Data Quality
- 🔄 Reproducibility
- 🔂 Recoverability
- 🔎 Auditability

### Learning Objectives

- Learn the basics constructs of Flyte: tasks, workflows, and launchplans
- Understand how Flyte orchestrates execution graphs, data, and compute infrastructure
- Work with the building blocks for productionizing data science workloads
- Learn how to test Flyte code, use CI/CD, and extend Flyte

## Outline

| 🔤 Introduction to Flyte | [45 minutes] |
| --- | --- |
| **Environment Setup** | Setting up your virtual development environment |
| **Flyte Basics** | Tasks, Workflows, and Launch Plans: the building blocks of Flyte |
| **`pyflyte run`** | Run tasks and workflows locally or on a Flyte cluster  |
| **Flyte Console** | A tour of the Flyte console to view workflow progress and status  |
| **`FlyteRemote`** | Programmatically run tasks and workflows  |
| **Scheduling Launch Plans** | Run your workflows on a schedule |

> ⏱️ 15 minute break

| 🔀 Flyte Programming Model | [45 minutes] |
| --- | --- |
| **Tasks as Containerized Functions** | A core building block for statelessness and reproducibility |
| **Workflows and Promises** | How Flyte workflows construct an execution graph of tasks |
| **Type System** | How Flyte handles different types |
| **How Data Flows in Flyte** | How the type system helps to abstract away passing data between tasks |
| **Lifecycle of a Workflow** | How workflows are executed by a Flyte backend |
| **Development Lifecycle Overview** | How to progress from development to production |

> ⏱️ 15 minute break

| 🚀 Productionizing Data Science Workloads | [45 mins] |
| --- | --- |
| **Parallelism** | Use dynamic workflows and map tasks to parallelize your tasks |
| **Horizontal Scaling** | Specify heterogenous resources requirements at the task-level and use plugins to horizontally scale your workloads |
| **Production Notebooks** | Use `NotebookTask`s to leverage jupyter notebooks in production workflows |
| **Container Tasks** | Incorporate tasks written in any language into your workflows |
| **Recovering from Failure** | Build robust pipelines wiht retries, caching, failure recovery, and checkpointing |
| **Auditing Workflows** | Gain visibility into your tasks with Flyte Decks |

> ⏱️ 15 minute break

| 🔎 Testing, CI/CD, Extending Flyte | [45 minutes] |
| --- | --- |
| **Writing Unit Tests** | Test Flyte tasks and workflows in isolation |
| **Writing Integration Tests** | Test Flyte workflows on a local cluster |
| **Using Github Actions** | Use github actions to automate testing |
| **Extending Flyte** | Extend Flyte with decorators, flyte deck extensions, type plugins, and task plugins |

> 🗣️ 15 minute Q&A

---
## Prerequisites

**⭐️ Important:** This workshop will involve moving between a terminal, text editor,
and jupyter notebook environment. We highly recommend using [VSCode](https://code.visualstudio.com/) for the purposes of this workshop, but you can use any combination
of tools that you're comfortable with.

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

#### Getting Help

🙌 Join the [flyte scipy slack channel](https://flyte-org.slack.com/archives/C05FQT6D26N)
to get help with anything from setup to debugging through out the workshop.

⏱️ We'll also be taking 15 minute breaks throughout the workshop if need to
catch up, ask any questions, or take a breather.

---

## Setup

Create a fork of this repo by going by going to the
[repo link](https://github.com/flyteorg/flyte-conference-talks) and clicking
on the **Fork** button on the top right of the page. Select your username as
the repo fork owner. This will result in a repository called
`https://github.com/<username>/flyte-conference-talks`, where `<username>` is your username.

Clone your fork of the repo (replace `<username>` with your actual username):

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
pip install -r requirements.txt flytekitplugins-envd
pip install jupyter
```

Test the virtual environment with:

```bash
pyflyte run \
    workflows/example_intro.py training_workflow \
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
export IMAGE=ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-latest

pyflyte run --remote \
    --image $IMAGE \
    workflows/example_intro.py training_workflow \
    --hyperparameters '{"C": 0.01}'
```


## Tests

Install dev dependencies:

```bash
pip install pytest pytest-xdist
source ~/venvs/scipy-2023/bin/activate
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
pytest tests/integration
```

> ℹ **Note**: Running the full integration test suite will take about 20 minutes.
> You can parallelize the test runner with by supplying the `pytest ... -n auto` flag.

## Troubleshooting

### `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.`

You may need to allow the default Docker socket to be used by third-party clients.
Enable this by going to the **Docker Desktop** application and navigating to:

`Settings > Advanced`

Then, click on the checkbox next to **Allow the default Docker socket to be used**,
then **Apply & restart**.

### `OOM Killed` error

In this case you may need to free up some memory by removing unused containers
with `docker system prune -a --volumes`.
