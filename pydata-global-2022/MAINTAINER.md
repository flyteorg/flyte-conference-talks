# For Maintainers

## Deployment

Build and push docker image:

```bash
./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v pydata-global-2022-v0
docker push ghcr.io/flyteorg/flyte-conference-talks:pydata-global-2022-v0
```

Serialize

```bash
pyflyte --pkgs workflows package --image ghcr.io/flyteorg/flyte-conference-talks:scipy-2022-v1 -f
```

Register

```bash
flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version v0
```

Fast Register

```bash
# fast serialize
pyflyte --pkgs workflows package --image ghcr.io/flyteorg/flyte-conference-talks:scipy-2022-v1 --fast -f

# fast register
flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version v0-fast0
```
