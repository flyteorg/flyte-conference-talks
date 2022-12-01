# Pydata Global 2022 Workshop

## Setup

Install venv

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

[Install Flytectl](https://docs.flyte.org/projects/flytectl/en/latest/#installation),
then start a Flyte demo cluster:

```
flytectl demo start --source .
```
