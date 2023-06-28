#!/bin/sh

pip install --no-cache-dir --user modin
pip install --no-cache-dir --user 'pyspark<3.4.0'
pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu torch
pip install --no-cache-dir --user pandas
pip install --no-cache-dir --user scikit-learn
pip install --no-cache-dir --user 'flytekit>=1.7.0'
pip install --no-cache-dir --user flytekitplugins-deck-standard
pip install --no-cache-dir --user flytekitplugins-pandera
pip install --no-cache-dir --user flytekitplugins-spark
pip install --no-cache-dir --user flytekitplugins-ray
pip install --no-cache-dir --user dataclasses_json joblib palmerpenguins wheel