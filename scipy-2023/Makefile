VERSION=$(shell git rev-parse HEAD | cut -c1-7)

docker-build:
	docker buildx build --push --tag ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-${VERSION} . --platform=linux/amd64,linux/arm64
	docker buildx build --push --tag ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-latest . --platform=linux/amd64,linux/arm64

ci-docker-build:
	docker build --push . -f Dockerfile.ci -t ghcr.io/flyteorg/flyte-conference-talks:scipy-2023-ci

sandbox-docker-build:
	docker build --push . -t localhost:30000/flyte-conference-talks:scipy-2023-ci --platform=linux/amd64,linux/arm64
