import numpy as np
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


@pytest.mark.parametrize(
    "predictions, input_ids, expected",
    [
        (
            np.array([[[True], [False]]]),  # 1 batch, 2 tokens, 1 label
            [101, 102],  # 2 tokens
            [[Annotation(id="0", tags={"CLASS_0"}, start=0, end=5, text="Hello")]],
        ),
        (
            np.array([[[True], [False]]]),
            [101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Add a lot of padding tokens
            [[Annotation(id="0", tags={"CLASS_0"}, start=0, end=5, text="Hello")]],
        ),
    ],
)
def test_map_prediction_to_tokens(annotator, predictions, input_ids, expected):
    # Mocking the batch encoding
    fake_batch_encoding = MagicMock()
    fake_batch_encoding.input_ids = input_ids

    # Mock tokenizer decode to return known strings to calculate offsets
    annotator.tokenizer.decode.side_effect = ["Hello", "World"]

    # Act
    mapped = annotator._map_prediction_to_tokens(predictions, [fake_batch_encoding])

    # Assert
    # We expect 1 annotation because only the first token was True
    for tested, expected in zip(mapped, expected):
        assert len(tested) == len(expected)
        for annotation, expected_annotation in zip(tested, expected):
            assert annotation.id == expected_annotation.id
            assert annotation.tags == expected_annotation.tags
            assert annotation.start == expected_annotation.start
            assert annotation.end == expected_annotation.end
            assert annotation.text == expected_annotation.text
