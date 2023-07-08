VERSION=$(shell git rev-parse HEAD | cut -c1-7)
docker-build-push:
	bash ./docker_build_push.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v scipy-2023-${VERSION}
	bash ./docker_build_push.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v scipy-2023-latest
