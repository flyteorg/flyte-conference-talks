from pathlib import Path
from flytekit.configuration import Config
from flytekit.remote import FlyteRemote, FlyteWorkflowExecution


def get_remote(local=None, config_file=None):
    return FlyteRemote(
        config=Config.auto(
            config_file=(
                None if local
                else config_file if config_file is not None
                else str(Path.home() / ".uctl" / "config.yaml")
            )
        ),
        default_project="onboarding",
        default_domain="development",
    )


def download_deck(
    remote: FlyteRemote,
    execution: FlyteWorkflowExecution,
    node_execution_key: str,
    local_path: str,
):
    exe = remote.sync_execution(execution=execution, sync_nodes=True)
    deck_uri = exe.node_executions[node_execution_key].closure.deck_uri
    response = remote.client.get_download_signed_url(deck_uri)
    remote.file_access.download(response.signed_url, local_path)
    print(f"Flyte decks for execution {execution.id.name} downloaded to {local_path}")
