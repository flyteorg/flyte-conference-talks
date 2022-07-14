# scipy_2022

A template for the recommended layout of a Flyte enabled repository for code written in python using [flytekit](https://docs.flyte.org/projects/flytekit/en/latest/).

## Usage

To get up and running with your Flyte project, we recommend following the
[Flyte getting started guide](https://docs.flyte.org/en/latest/getting_started.html).


## Deployment

Create project on Flyte cluster:

```bash
flytectl create project --project scipy-2022 --name scipy-2022 --id scipy-2022 --description 'workflow examples for scipy 2022 talk' --config ~/.flyte/unionplayground-config.yaml
```

Build and push docker image:

```bash
./docker_build_and_tag.sh -r ghcr.io/flyteorg -a flyte-conference-talks -v scipy-2022-v1
docker push ghcr.io/flyteorg/flyte-conference-talks:scipy-2022-v1
```

Serialize

```bash
pyflyte --pkgs workflows package --image ghcr.io/flyteorg/flyte-conference-talks:scipy-2022-v1 -f
```

Register

```bash
flytectl register files -c ~/.flyte/unionplayground-config.yaml --project scipy-2022 --domain development --archive flyte-package.tgz --version v1
```

Fast Register

```bash
# fast serialize
pyflyte --pkgs workflows package --image ghcr.io/flyteorg/flyte-conference-talks:scipy-2022-v1 --fast -f

# fast register
flytectl register files --project scipy-2022 --domain development --archive flyte-package.tgz --version v1-fast1
```
