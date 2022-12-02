# this is needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	pip install --no-cache-dir --user modin && \
	pip install --no-cache-dir --user pyspark && \
	pip install --no-cache-dir --user torch && \
	pip install --no-cache-dir --user pandas && \
	pip install --no-cache-dir --user scikit-learn && \
	pip install --no-cache-dir --user flytekit>=1.2.4 flytekitplugins-deck-standard flytekitplugins-pandera flytekitplugins-spark flytekitplugins-ray && \
	pip install --no-cache-dir --user -r requirements.txt && \
	pip install --no-cache-dir --user -e .
