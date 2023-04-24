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
	bash ./docker_build_and_tag.sh -r ghcr.io/eapolinario -a flyte-conference-talks -v pydata-seattle-2023-${VERSION}
	# bash ./docker_build_and_tag.sh -r ghcr.io/eapolinario -a flyte-conference-talks -v pydata-seattle-2023-latest
	# docker push ghcr.io/eapolinario/flyte-conference-talks:pydata-seattle-2023-${VERSION}
	# docker push ghcr.io/eapolinario/flyte-conference-talks:pydata-seattle-2023-latest
