# this is a hack, needed because of limited memory on Flyte Sandbox server
sandbox-setup:
	bash setup.sh && bash setup.sh


VERSION = "$(shell git rev-parse HEAD)"


docker-build-push:
	./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-global-2022-${VERSION}
	./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-global-2022-latest
	docker push ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-${VERSION}
	docker push ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-latest
