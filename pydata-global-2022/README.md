# Flyte Tutorial: Pydata Global 2022

This directory contains the workshop materials for [Pydata global 2022](https://pydata.org/global2022/).

## Production-grade Machine Learning with Flyte

### Abstract

As the discipline of machine learning Operations (MLOps) matures, it’s becoming
clear that, in practice, building ML models poses additional challenges compared
to the traditional software development lifecycle. This tutorial will focus on
four challenges in the context of ML model development:

- Scalability
- Data quality
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
- **Randomness and Resource Requirements**: Code- and compute-level reproducibility.

### Recoverability [15 minutes]

- **Caching**: Don't waste precious compute re-running nodes.
- **Recovering Executions**: Don't waste precious compute re-running nodes.
- **Checkpointing**: Checkpoint progress within a node.

### Auditability [10 minutes]

- **Flyte Decks**: Create rich static reports associated with your data/model artifacts.
- **Extending Flyte Decks**: Write your own Flyte Deck visualizations.


## Setup

Follow the **Flyte Sandbox** instructions to use the fully-managed prototyping
environment hosted by [Union.ai](https://www.union.ai/), or the **Local**
instructions if you want to run a Flyte cluster on your local machine.

### Flyte Sandbox

Go to https://sandbox.union.ai and create an account via Github or Google. Then,
click **Start Sandbox**. This will take 1-2 minutes to launch.

<img src="static/flyte_sandbox_start.png" alt="flyte sandbox start" width="600"/>


When the sandbox is ready, click **Code Editor**, which will take you to a VSCode IDE.

<img src="static/flyte_sandbox_ready.png" alt="flyte sandbox ready" width="600"/>

In the terminal, clone this repo and go to the workshop directory:

```
git clone https://github.com/flyteorg/flyte-conference-talks
cd flyte-conference-talks/pydata-global-2022
```

Install dependencies with

```
make sandbox-setup
```

To make sure everything's working, run

```
python workflows/example_00_intro.py
```

Expected output:

```
DefaultNamedTupleOutput(o0=LogisticRegression(C=0.1, max_iter=5000.0), o1=0.989010989010989, o2=1.0)
```

### Local

Clone this repo and go to the workshop directory:

```
git clone https://github.com/flyteorg/flyte-conference-talks
cd flyte-conference-talks/pydata-global-2022
```

Create a virtual environment

```
python -m venv ~/venvs/pydata-global-2022
source ~/venvs/pydata-global-2022/bin/activate
```

Install dependencies

```
pip install -e .
```

[Install Flytectl](https://docs.flyte.org/projects/flytectl/en/latest/#installation),
then start a Flyte demo cluster:

```
flytectl demo start --source .
```

Then build the docker image for the tutorial inside the cluster:

```
export IMAGE=ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-latest
flytectl demo exec -- docker build . --tag $IMAGE
```
