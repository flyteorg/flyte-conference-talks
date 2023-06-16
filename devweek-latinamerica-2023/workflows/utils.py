import os
from flytekit.configuration import Config
from flytekit.remote import FlyteRemote


def get_remote(local=None):
    if local is None:
        is_sandbox = (
            os.environ.get("UNION_SANDBOX_PLACEHOLDER_SERVICE_HOST") is not None
        )
    else:
        is_sandbox = not local
    return FlyteRemote(
        config=Config.auto(config_file=None if is_sandbox else "./config.yaml"),
        default_project="flytesnacks",
        default_domain="development",
    )
