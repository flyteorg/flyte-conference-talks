# this is a hack, needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	bash setup.sh
	flytectl update task-resource-attribute --attrFile cra.yaml

local-sandbox-setup:
	flytectl demo start --source .
	bash local-setup.sh
	sleep 1
	flytectl update task-resource-attribute --attrFile cra.yaml


VERSION=$(shell git rev-parse HEAD)
docker-build-push:
	bash ./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v scipy-2023-${VERSION}
	bash ./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v scipy-2023-latest
