import pytest
import json
import xml.etree.ElementTree as ET

from spesia_ner.data_models import (
    AgentAnnotation,
    AgentAnnotationsList,
    Annotation,
    Record,
)


# --- FIXTURES ---


@pytest.fixture
def sample_xml_content():
    """Provides the XML structure needed for from_xml testing."""
    return """
    <ROOT>
        <TEXT>The quick brown fox jumps over the lazy dog.</TEXT>
        <TAGS>
            <TAG id="T1" tag="ANIMAL|MAMMAL" start="4" end="15" />
            <TAG id="T2" tag="ANIMAL" start="35" end="43" />
        </TAGS>
    </ROOT>
    """


@pytest.fixture
def sample_argilla_json_content():
    """Provides the nested JSON structure for from_argilla_json testing."""
    return {
        "record-001": {
            "fields": {"text": "Patient has lung cancer."},
            "responses": {
                "annotations": [
                    {  # User 1
                        "value": [
                            {"start": 12, "end": 23, "label": "DISEASE"},
                            {
                                "start": 12,
                                "end": 23,
                                "label": "ONCOLOGY",
                            },  # Multi-tag from same span
                        ]
                    }
                ]
            },
        },
        "record-002": {
            "fields": {"text": "Next patient is fine."},
            "responses": {"annotations": []},
        },
    }


@pytest.fixture
def sample_doccano_jsonl_content():
    """Provides the Doccano JSONL structure for from_doccano_jsonl testing."""
    return [
        {
            "text": "Doccano starts index at 1.",
            "labels": [
                [15, 23, "INDEX"],
                [24, 25, "ONE_CHAR"],
            ],
        },
        {
            "text": "Another document.",
            "entities": [{"start": 8, "end": 16, "label": "DOC"}],
        },
    ]


@pytest.fixture
def sample_record():
    """Provides a basic Record instance for testing properties and conversions."""
    return Record(
        text="Sample text with tags and groups.",
        annotations=[
            Annotation(
                id="A1",
                tags={"PERSON", "NOUN"},
                semantic_groups={"G1"},
                start=0,
                end=6,
                text="Sample",
            ),
            Annotation(
                id="A2",
                tags={"TIME"},
                semantic_groups={"G2"},
                start=12,
                end=16,
                text="text",
            ),
        ],
    )


# --- TEST SUITE ---

## Testing Properties and Conversion Logic


def test_record_tags_property(sample_record):
    """Checks that the .tags property aggregates all unique tags."""
    # Arrange: sample_record is ready
    # Act
    all_tags = sample_record.tags
    # Assert
    assert all_tags == {"PERSON", "NOUN", "TIME"}


def test_record_semantic_groups_property(sample_record):
    """Checks that the .semantic_groups property aggregates all unique groups."""
    # Arrange: sample_record is ready
    # Act
    all_groups = sample_record.semantic_groups
    # Assert
    assert all_groups == {"G1", "G2"}


@pytest.mark.parametrize(
    "label_type, expected_labels",
    [
        ("tags", {"PERSON", "NOUN", "TIME"}),
        ("semantic_groups", {"G1", "G2"}),
    ],
)
def test_to_agent_annotations_conversion(sample_record, label_type, expected_labels):
    """Verifies that to_agent_annotations flattens tags/groups correctly."""
    # Act
    agent_list = sample_record.to_agent_annotations(label_type)

    # Assert
    # Total annotations should be the sum of tags/groups across all annotations
    assert len(agent_list.annotations) == len(expected_labels)

    # Check if all expected labels are present
    actual_labels = {ann.label for ann in agent_list.annotations}
    assert actual_labels == expected_labels

    # Check that text is preserved and correctly duplicated
    texts = [ann.text for ann in agent_list.annotations]
    assert all(t in ["Sample", "text"] for t in texts)


def test_from_agent_annotations_to_record():
    """Tests reverse conversion and character indexing logic (text.find)."""
    # Arrange
    text = "The medication is Tylenol. The patient has fever."
    agent_list = AgentAnnotationsList(
        annotations=[
            AgentAnnotation(label="DRUG", text="Tylenol"),
            AgentAnnotation(label="SYMPTOM", text="fever"),
            AgentAnnotation(
                label="MISSED", text="Medication"
            ),  # Should be skipped in text.find
            AgentAnnotation(
                label="DUPLICATE", text="The"
            ),  # Should find first occurrence only (index 0)
        ]
    )

    # Act
    record_tags = Record.from_agent_annotations(text, agent_list, label_type="tags")
    record_groups = Record.from_agent_annotations(
        text, agent_list, label_type="semantic_groups"
    )

    # Assert
    assert len(record_tags.annotations) == 3  # 'medication' is not found by text.find

    print(record_tags)

    # 1. Tags check
    tylenol_ann = [a for a in record_tags.annotations if a.text == "Tylenol"][0]
    assert tylenol_ann.tags == {"DRUG"}
    assert tylenol_ann.semantic_groups == set()
    assert tylenol_ann.start == 18
    assert tylenol_ann.end == 25

    # 2. Semantic Groups check (should be mutually exclusive)
    tylenol_group_ann = [a for a in record_groups.annotations if a.text == "Tylenol"][0]
    assert tylenol_group_ann.tags == set()
    assert tylenol_group_ann.semantic_groups == {"DRUG"}

    # 3. First occurrence check
    the_ann = [a for a in record_tags.annotations if a.text == "The"][0]
    assert the_ann.start == 0


# --- Testing File Parsing (I/O) ---

## Testing from_xml


def test_record_from_xml_success(tmp_path, sample_xml_content):
    """Tests parsing a standard XML file with character offset checks."""
    # Arrange
    xml_file = tmp_path / "test.xml"
    # Write the content using ElementTree to ensure valid XML structure
    root = ET.fromstring(sample_xml_content)
    tree = ET.ElementTree(root)
    tree.write(xml_file)

    # Act
    record = Record.from_xml(xml_file)

    # Assert
    assert record.text == "The quick brown fox jumps over the lazy dog."
    assert len(record.annotations) == 2

    ann1 = record.annotations[0]
    assert ann1.tags == {"ANIMAL", "MAMMAL"}
    assert ann1.start == 4
    assert ann1.end == 15
    assert ann1.text == "quick brown"

    ann2 = record.annotations[1]
    assert ann2.tags == {"ANIMAL"}
    assert ann2.start == 35
    assert ann2.end == 43
    assert ann2.text == "lazy dog"


def test_record_from_xml_string_path(tmp_path, sample_xml_content):
    """Tests that the function handles a string input path correctly."""
    # Arrange
    xml_file = tmp_path / "test_str.xml"
    tree = ET.ElementTree(ET.fromstring(sample_xml_content))
    tree.write(xml_file)

    # Act
    record = Record.from_xml(str(xml_file))

    # Assert
    assert record.text == "The quick brown fox jumps over the lazy dog."


## Testing from_argilla_json


def test_record_from_argilla_json_success(tmp_path, sample_argilla_json_content):
    """Tests parsing the Argilla JSON format, including multi-tagging and user separation."""
    # Arrange
    json_file = tmp_path / "argilla.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(sample_argilla_json_content, f)

    # Act
    records = Record.from_argilla_json(json_file)

    # Assert
    assert (
        len(records) == 1
    )  # Only one record has annotations, and one user annotated it.

    record = records[0]
    assert record.text == "Patient has lung cancer."
    assert (
        len(record.annotations) == 2
    )  # Two annotations, one for 'DISEASE', one for 'ONCOLOGY'

    # Check annotation 1
    ann1 = [a for a in record.annotations if "DISEASE" in a.tags][0]
    assert ann1.tags == {"DISEASE"}
    assert ann1.start == 12
    assert ann1.end == 23
    assert ann1.text == "lung cancer"

    # Check annotation 2 (same span, different tag)
    ann2 = [a for a in record.annotations if "ONCOLOGY" in a.tags][0]
    assert ann2.tags == {"ONCOLOGY"}
    assert ann2.text == "lung cancer"


## Testing from_doccano_jsonl


def test_record_from_doccano_jsonl_success(tmp_path, sample_doccano_jsonl_content):
    """Tests parsing Doccano JSONL, handling different formats and index offsets."""
    # Arrange
    jsonl_file = tmp_path / "doccano.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in sample_doccano_jsonl_content:
            f.write(json.dumps(item) + "\n")

    # Act
    records = Record.from_doccano_jsonl(jsonl_file)

    # Assert
    assert len(records) == 2

    # Check Record 1 (list format with index fix)
    record1 = records[0]
    assert record1.text == "Doccano starts index at 1."
    assert len(record1.annotations) == 2

    ann1 = record1.annotations[0]
    assert ann1.tags == {"INDEX"}
    assert ann1.start == 15
    assert ann1.end == 23
    assert ann1.text == "index at"

    # Check Record 2 (dict format)
    record2 = records[1]
    assert record2.text == "Another document."
    assert len(record2.annotations) == 1
    ann3 = record2.annotations[0]
    assert ann3.tags == {"DOC"}
    assert ann3.start == 8
    assert ann3.end == 16
    assert ann3.text == "document"


## Testing from_file (The Dispatcher)


def test_from_file_xml_dispatch(tmp_path, mocker):
    """Tests if from_file correctly calls from_xml for .xml files."""
    # Arrange
    xml_file = tmp_path / "data.xml"
    xml_file.touch()  # Create an empty file

    # Mock the internal methods to confirm which one is called
    mock_from_xml = mocker.patch.object(Record, "from_xml", return_value=[])
    mock_from_argilla_json = mocker.patch.object(
        Record, "from_argilla_json", return_value=[]
    )
    mock_from_doccano_jsonl = mocker.patch.object(
        Record, "from_doccano_jsonl", return_value=[]
    )

    # Act
    Record.from_file(xml_file)

    # Assert
    mock_from_xml.assert_called_once_with(xml_file)
    mock_from_argilla_json.assert_not_called()
    mock_from_doccano_jsonl.assert_not_called()


def test_from_file_jsonl_dispatch(tmp_path, mocker):
    """Tests if from_file correctly calls from_doccano_jsonl for .jsonl files."""
    # Arrange
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.touch()

    # Mock the internal methods
    mock_from_doccano_jsonl = mocker.patch.object(
        Record, "from_doccano_jsonl", return_value=[]
    )

    # Act
    Record.from_file(jsonl_file)

    # Assert
    mock_from_doccano_jsonl.assert_called_once_with(jsonl_file)


def test_from_file_unsupported_format(tmp_path):
    """Tests that an unsupported file format raises the expected ValueError."""
    # Arrange
    bad_file = tmp_path / "data.txt"
    bad_file.touch()

    # Act / Assert
    with pytest.raises(ValueError, match="Unsupported file format"):
        Record.from_file(bad_file)
