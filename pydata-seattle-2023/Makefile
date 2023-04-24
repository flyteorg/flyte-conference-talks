# this is a hack, needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	bash setup.sh
	flytectl update task-resource-attribute --attrFile cra.yaml

local-sandbox-setup:
	flytectl demo start --source .
	bash local-setup.sh
	flytectl update task-resource-attribute --attrFile cra.yaml


VERSION = "$(shell git rev-parse HEAD)"


docker-build-push:
	./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-global-2022-${VERSION}
	./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-global-2022-latest
	docker push ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-${VERSION}
	docker push ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-latest
