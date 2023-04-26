#!/bin/sh

pip install --no-cache-dir --user modin
pip install --no-cache-dir --user pyspark
pip install --no-cache-dir --user torch
pip install --no-cache-dir --user pandas
pip install --no-cache-dir --user scikit-learn
pip install --no-cache-dir --user 'flytekit>=1.2.4'
pip install --no-cache-dir --user flytekitplugins-deck-standard
pip install --no-cache-dir --user flytekitplugins-pandera
pip install --no-cache-dir --user flytekitplugins-spark
pip install --no-cache-dir --user flytekitplugins-ray
pip install --no-cache-dir --user -r requirements.txt
