import pytest
from unittest.mock import MagicMock, patch

from spesia_ner.autolabeling import AutoAnnotator
from spesia_ner.data_models import Annotation


# Define fixtures
@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock()
    return model


@pytest.fixture
def annotator(mock_tokenizer, mock_model):
    """Creates an instance of the AutoAnnotator class with mocked dependencies."""

    idx_to_label = {0: "CLASS_0", 1: "CLASS_1"}
    thresholds = [0.5, 0.5]

    return AutoAnnotator(
        tokenizer=mock_tokenizer,
        model=mock_model,
        best_thresholds=thresholds,
        idx_to_label=idx_to_label,
        batch_size=2,
    )


def test_annotations_merge(annotator):
    # Arrange
    raw_annotations = [
        Annotation(id="", tags={"DISEASE"}, start=0, end=4, text="Lung"),
        Annotation(
            id="", tags={"DISEASE"}, start=4, end=10, text=" Cancer"
        ),  # Sequential
        Annotation(id="", tags={"OTHER"}, start=15, end=20, text=" unrelated"),
        Annotation(id="", tags={"DISEASE"}, start=20, end=24, text="Lung"),  # Gap
    ]

    # Act
    merged = annotator._merge_annotations(raw_annotations)

    # Assert
    assert len(merged) == 3

    # Verify the merged entity
    first_lung_cancer = merged[0]
    assert first_lung_cancer.start == 0
    assert first_lung_cancer.end == 10
    assert first_lung_cancer.tags == {"DISEASE"}

    second_lung_cancer = merged[2]
    assert second_lung_cancer.start == 20
    assert second_lung_cancer.end == 24
    assert second_lung_cancer.tags == {"DISEASE"}
