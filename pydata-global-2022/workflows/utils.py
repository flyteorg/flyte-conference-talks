from flytekit.remote import FlyteRemote, FlyteWorkflowExecution


def download_deck(
    remote: FlyteRemote,
    execution: FlyteWorkflowExecution,
    local_path: str,
):
    remote.file_access.download(execution.node_executions["n1"].closure.deck_uri, local_path)
    print(f"Flyte decks for execution {execution.id.name} downloaded to {local_path}")
