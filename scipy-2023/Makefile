# this is a hack, needed because of limited memory on Flyte Sandbox server
remote-sandbox-setup:
	bash setup.sh
	bash setup.sh
	flytectl update task-resource-attribute --attrFile cra.yaml

sandbox-setup:
	flytectl demo start --source .
	sleep 1
	FLYTECTL_CONFIG=~/.flyte/config-sandbox.yaml flytectl update task-resource-attribute --attrFile cra.yaml


VERSION=$(shell git rev-parse HEAD | cut -c1-7)
docker-build-push:
	bash ./docker_build_push.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v scipy-2023-${VERSION}
	bash ./docker_build_push.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v scipy-2023-latest
