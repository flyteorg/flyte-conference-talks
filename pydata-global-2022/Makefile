# this is needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	pip install --no-cache-dir --user modin && \
	pip install --no-cache-dir --user pyspark && \
	pip install --no-cache-dir --user torch && \
	pip install --no-cache-dir --user pandas && \
	pip install --no-cache-dir --user scikit-learn && \
	pip install --no-cache-dir --user 'flytekit>=1.2.4' && \
	pip install --no-cache-dir --user flytekitplugins-deck-standard && \
	pip install --no-cache-dir --user flytekitplugins-pandera && \
	pip install --no-cache-dir --user flytekitplugins-spark && \
	pip install --no-cache-dir --user flytekitplugins-ray && \
	pip install --no-cache-dir --user -r requirements.txt && \
	pip install --no-cache-dir --user -e .


VERSION = "$(shell git rev-parse HEAD)"


docker-build-push:
	./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-global-2022-${VERSION}
	./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-global-2022-latest
	docker push ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-${VERSION}
	docker push ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-latest
