"""Auditability: Extending Flyte Decks."""

import base64
from dataclasses import asdict
from io import BytesIO

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._estimator_html_repr import estimator_html_repr

from flytekit import task, workflow, Deck

from workflows.example_00_intro import get_data, split_data, FEATURES, TARGET
from workflows.example_06_reproducibility import Hyperparameters


class SklearnEstimatorRenderer:
    """🃏 Easily extend Flyte Decks to visualize our model pipeline"""

    def to_html(self, pipeline: Pipeline) -> str:
        return estimator_html_repr(pipeline)


@task(disable_deck=False)
def train_model(
    data: pd.DataFrame, hyperparameters: Hyperparameters
) -> Pipeline:
    pipeline = Pipeline(
        [
            ("preprocessing", StandardScaler()),
            ("classifier", SGDClassifier(**asdict(hyperparameters))),
        ]
    )
    Deck("model_pipeline", SklearnEstimatorRenderer().to_html(pipeline))
    return pipeline.fit(data[FEATURES], data[TARGET])


class ConfusionMatrixRenderer:

    def to_html(self, cm_display: ConfusionMatrixDisplay) -> str:
        buf = BytesIO()
        cm_display.plot().figure_.savefig(buf, format="png")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"<img src='data:image/png;base64,{encoded}'>"


@task(disable_deck=False)
def evaluate(model: Pipeline, data: pd.DataFrame, split: str):
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            data[TARGET], model.predict(data[FEATURES])
        ),
        display_labels=model.named_steps["classifier"].classes_,
    )
    Deck(f"evaluation {split}", ConfusionMatrixRenderer().to_html(cm_display))


@workflow
def training_workflow(
    hyperparameters: Hyperparameters,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    data = get_data()
    train_data, test_data = split_data(
        data=data, test_size=test_size, random_state=random_state
    )
    model = train_model(data=train_data, hyperparameters=hyperparameters)
    evaluate(model=model, data=train_data, split="train")
    evaluate(model=model, data=test_data, split="test")
    return model


if __name__ == "__main__":
    hyperparameters = Hyperparameters(
        penalty="l1", alpha=0.03, random_state=12345
    )
    print(f"{training_workflow(hyperparameters=hyperparameters)}")
