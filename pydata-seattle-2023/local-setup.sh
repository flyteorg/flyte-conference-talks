#!/bin/sh

python -m venv .venv
. .venv/bin/activate

VERSION=1.2.4
pip install jupyter modin pyspark torch pandas scikit-learn \
    flytekit==$VERSION flytekitplugins-deck-standard==$VERSION flytekitplugins-pandera==$VERSION flytekitplugins-spark==$VERSION flytekitplugins-ray==$VERSION
pip install -e .
