# this is a hack, needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	bash setup.sh
	flytectl update task-resource-attribute --attrFile cra.yaml

local-sandbox-setup:
	flytectl demo start --source .
	bash local-setup.sh
	flytectl update task-resource-attribute --attrFile cra.yaml


docker-build-push:
	VERSION=$(shell git rev-parse HEAD)
	bash ./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-seattle-2023-${VERSION}
	bash ./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-seattle-2023-latest
