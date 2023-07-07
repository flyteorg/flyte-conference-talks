# Flyte Tutorial: Scipy 2023

This directory contains the workshop materials for [Scipy 2023](https://cfp.scipy.org/2023/talk/YHEYVY/).

## Production-grade Machine Learning with Flyte

### Abstract

As the discipline of machine learning Operations (MLOps) matures, it’s becoming
clear that, in practice, building ML models poses additional challenges compared
to the traditional software development lifecycle. This tutorial will focus on
four challenges in the context of ML model development:

- Scalability
- Data Quality
- Reproducibility
- Recoverability
- Auditability

Using Flyte, a data- and machine-learning-aware open source workflow orchestrator,
we’ll see how to address these challenges and abstract them out to give you a
broader understanding of how to surmount them.

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

### Introduction [20 minutes]

- **Environment Setup**: Setting up your Flyte Sandbox environment.
- **Flyte Basics**: The building blocks for expressing execution graphs.

### Scalability [15 minutes]

- **Dynamic Workflows**: Defining execution graphs at runtime.
- **Map Tasks**: Scale embarrassingly parallel workflows.
- **Declarative Resource Allocation and Plugins**: SQL, Spark, and Ray task plugins.

### Data Quality [10 minutes]

- **Type System**: Understand the benefits of static type safety.
- **DataFrame Types**: Validate tabular data at runtime.

### Break [10 minutes]

### Reproducibility [10 minutes]

- **Containerization**: Containerize your workflows for dependency isolation.
- **Randomness and Resource Requirements**: Code- and resource-level reproducibility.

### Recoverability [15 minutes]

- **Caching**: Don't waste precious compute re-running nodes.
- **Recovering Executions**: Don't waste precious compute re-running nodes.
- **Checkpointing**: Checkpoint progress within a node.

### Auditability [10 minutes]

- **Flyte Decks**: Create rich static reports associated with your data/model artifacts.
- **Extending Flyte Decks**: Write your own Flyte Deck visualizations.


## Prerequisites

> ⚠️ NOTE: Windows users need to have [WSL installed](https://docs.microsoft.com/en-us/windows/wsl/install-win10) in order to run this workshop.

Install [Docker Desktop](https://docs.docker.com/get-docker/) and make sure the
Docker daemon is running.

Install `flytectl`:

```bash
# Homebrew (MacOS)
brew install flyteorg/homebrew-tap/flytectl

# Or Curl
curl -sL https://ctl.flyte.org/install | sudo bash -s -- -b /usr/local/bin
```

## Setup

Clone this repo and go to the workshop directory:

```bash
git clone https://github.com/flyteorg/flyte-conference-talks
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

Start the local Flyte sandbox:

```bash
flytectl demo start --source .
FLYTECTL_CONFIG=~/.flyte/config-sandbox.yaml

# update task resources
flytectl update task-resource-attribute --attrFile cra.yaml
```

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
