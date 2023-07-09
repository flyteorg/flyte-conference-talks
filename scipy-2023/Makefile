VERSION=$(shell git rev-parse HEAD | cut -c1-7)

docker-build:
	docker build --push --tag ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-${VERSION} .
	docker build --push --tag ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-latest .

ci-docker-build:
	docker build --push . -f Dockerfile.ci -t ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-ci
