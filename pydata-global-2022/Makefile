# this is needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	pip install --no-cache-dir modin && \
	pip install --no-cache-dir pyspark && \
	pip install --no-cache-dir torch && \
	pip install --no-cache-dir flytekit>=1.2.4 flytekitplugins-deck-standard flytekitplugins-pandera flytekitplugins-spark flytekitplugins-ray && \
	pip install --no-cache-dir pandas && \
	pip install --no-cache-dir scikit-learn && \
	pip install --no-cache-dir -r requirements.txt
