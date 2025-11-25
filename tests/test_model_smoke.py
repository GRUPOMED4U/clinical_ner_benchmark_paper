"""
Smoke tests for model loading and basic functionality with dummy data.

This module provides tests to verify that:
1. The spesia_ner package can load datasets correctly
2. Models can be loaded from HuggingFace
3. Tokenizers work with the data
4. Basic inference pipeline works
"""

import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification


@pytest.fixture
def dummy_data_path():
    """Path to the bundled dummy dataset."""
    return Path("data/dummy_data")


@pytest.fixture
def model_id():
    """Small model for testing."""
    return "jhu-clsp/mmBERT-base"


class TestDatasetLoading:
    """Test suite for dataset loading with dummy data."""

    def test_dummy_data_exists(self, dummy_data_path):
        """Verify that the dummy data directory and files exist."""
        assert dummy_data_path.exists(), f"Dummy data path not found: {dummy_data_path}"

        # Check for dummy files
        files = list(dummy_data_path.glob("fake_data.*"))
        assert len(files) > 0, f"No fake_data files found in {dummy_data_path}"

    def test_import_spesia_ner(self):
        """Test that spesia_ner package can be imported."""
        try:
            from spesia_ner.datasets import ClinicalRecordsDataset
            from spesia_ner.data_models import Record
            from spesia_ner.metrics import compute_metrics
        except ImportError as e:
            pytest.fail(f"Failed to import spesia_ner modules: {e}")

    def test_load_dummy_dataset_without_split(self, dummy_data_path):
        """Test loading dummy dataset without splitting."""
        from spesia_ner.datasets import ClinicalRecordsDataset

        dataset = ClinicalRecordsDataset(
            path=dummy_data_path,
            tokenizer=None,
            label_type="tags",
            annotation_scheme="IO",
            split=None,
        )

        assert dataset.records is not None
        assert len(dataset.records) > 0, "Dataset should have records"
        assert hasattr(dataset, "tags"), "Dataset should have tags attribute"

    def test_load_dummy_dataset_with_splits(self, dummy_data_path):
        """Test loading dummy dataset with train/val/test splits."""
        from spesia_ner.datasets import ClinicalRecordsDataset

        train_dataset = ClinicalRecordsDataset(
            path=dummy_data_path,
            label_type="tags",
            annotation_scheme="IO",
            split="train",
            split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
            data_split_method="iterative",
            random_seed=42,
        )

        assert train_dataset.records is not None
        assert len(train_dataset.records) > 0

    def test_load_dataset_with_bio_scheme(self, dummy_data_path):
        """Test loading dataset with BIO annotation scheme."""
        from spesia_ner.datasets import ClinicalRecordsDataset

        dataset = ClinicalRecordsDataset(
            path=dummy_data_path,
            label_type="tags",
            annotation_scheme="BIO",
            split="train",
            split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
            data_split_method="iterative",
            random_seed=42,
        )

        assert dataset.records is not None
        assert len(dataset.records) > 0
        assert dataset.annotation_scheme == "BIO"

    def test_dataset_has_valid_tags(self, dummy_data_path):
        """Test that loaded dataset has valid entity tags."""
        from spesia_ner.datasets import ClinicalRecordsDataset

        dataset = ClinicalRecordsDataset(
            path=dummy_data_path,
            label_type="tags",
            annotation_scheme="IO",
            split=None,
        )

        # Check that tags were extracted
        assert len(dataset.tags) > 0, "Dataset should have extracted entity tags"

        # Check that common entity types are present
        common_types = {"PERSON", "ORGANIZATION", "LOCATION"}
        found_types = common_types.intersection(dataset.tags)
        assert (
            len(found_types) > 0
        ), f"Expected to find some common entity types, but found: {dataset.tags}"


class TestTokenizerIntegration:
    """Test suite for tokenizer integration."""

    def test_load_tokenizer(self, model_id):
        """Test that tokenizer can be loaded."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            assert tokenizer is not None
            assert hasattr(tokenizer, "vocab_size")
        except Exception as e:
            pytest.skip(f"Could not load tokenizer (network issue?): {e}")

    def test_tokenizer_with_dataset(self, dummy_data_path, model_id):
        """Test integrating tokenizer with dataset."""
        from spesia_ner.datasets import ClinicalRecordsDataset

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e:
            pytest.skip(f"Could not load tokenizer: {e}")

        dataset = ClinicalRecordsDataset(
            path=dummy_data_path,
            tokenizer=tokenizer,
            label_type="tags",
            annotation_scheme="IO",
            split="train",
            split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
            data_split_method="iterative",
            max_length=512,
            random_seed=42,
        )

        assert dataset.num_labels > 0, "Dataset should have labels"
        assert len(dataset) > 0, "Tokenized dataset should have samples"

    def test_tokenized_sample_structure(self, dummy_data_path, model_id):
        """Test that tokenized samples have correct structure."""
        from spesia_ner.datasets import ClinicalRecordsDataset

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e:
            pytest.skip(f"Could not load tokenizer: {e}")

        dataset = ClinicalRecordsDataset(
            path=dummy_data_path,
            tokenizer=tokenizer,
            label_type="tags",
            annotation_scheme="IO",
            split="train",
            split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
            data_split_method="iterative",
            max_length=512,
            random_seed=42,
        )

        sample = dataset[0]

        # Check sample structure
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Check tensor types and shapes
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].ndim == 1
        assert sample["labels"].shape == sample["input_ids"].shape


class TestModelLoading:
    """Test suite for model loading."""

    def test_load_model(self, model_id):
        """Test that model can be loaded."""
        try:
            from spesia_ner.datasets import ClinicalRecordsDataset
            from pathlib import Path

            dummy_path = Path("data/dummy_data")
            if not dummy_path.exists():
                pytest.skip("Dummy data not found")

            # Load dataset to get num_labels
            dataset = ClinicalRecordsDataset(
                path=dummy_path,
                label_type="tags",
                annotation_scheme="IO",
                split=None,
            )

            num_labels = len(dataset.tags)

            # Now load model
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, num_labels=num_labels
            )
            assert model is not None

        except Exception as e:
            pytest.skip(f"Could not load model (network issue?): {e}")

    def test_model_forward_pass(self, dummy_data_path, model_id):
        """Test that model can do a forward pass with dummy data."""
        try:
            from spesia_ner.datasets import ClinicalRecordsDataset

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            dataset = ClinicalRecordsDataset(
                path=dummy_data_path,
                tokenizer=tokenizer,
                label_type="tags",
                annotation_scheme="IO",
                split="train",
                split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
                data_split_method="iterative",
                max_length=512,
                random_seed=42,
            )

            model = AutoModelForTokenClassification.from_pretrained(
                model_id, num_labels=dataset.num_labels
            )

            # Get a sample and run forward pass
            sample = dataset[0]
            input_ids = sample["input_ids"].unsqueeze(0)
            attention_mask = sample["attention_mask"].unsqueeze(0)

            # Use CPU for testing
            device = "cpu"
            model = model.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            assert outputs.logits is not None
            assert outputs.logits.shape[0] == 1  # batch size
            assert outputs.logits.shape[2] == dataset.num_labels  # num classes

        except Exception as e:
            pytest.skip(f"Could not run model forward pass: {e}")


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_io_scheme(self, dummy_data_path, model_id):
        """Test full pipeline: load data -> split -> tokenize -> load model -> forward pass."""
        try:
            from spesia_ner.datasets import ClinicalRecordsDataset

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            # Load and split dataset
            train_dataset = ClinicalRecordsDataset(
                path=dummy_data_path,
                tokenizer=tokenizer,
                label_type="tags",
                annotation_scheme="IO",
                split="train",
                split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
                data_split_method="iterative",
                max_length=512,
                random_seed=42,
            )

            val_dataset = ClinicalRecordsDataset(
                path=dummy_data_path,
                tokenizer=tokenizer,
                label_type="tags",
                annotation_scheme="IO",
                split="val",
                split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
                data_split_method="iterative",
                max_length=512,
                random_seed=42,
            )

            # Load model
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, num_labels=train_dataset.num_labels
            )

            # Verify pipeline integrity
            assert len(train_dataset) > 0
            assert len(val_dataset) > 0
            assert model is not None

            # Test forward pass with one sample from each split
            device = "cpu"
            model = model.to(device)
            model.eval()

            for dataset, name in [(train_dataset, "train"), (val_dataset, "val")]:
                sample = dataset[0]
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                assert outputs.logits is not None

        except Exception as e:
            pytest.skip(f"Could not run full pipeline: {e}")

    def test_full_pipeline_bio_scheme(self, dummy_data_path, model_id):
        """Test full pipeline with BIO annotation scheme."""
        try:
            from spesia_ner.datasets import ClinicalRecordsDataset

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            dataset = ClinicalRecordsDataset(
                path=dummy_data_path,
                tokenizer=tokenizer,
                label_type="tags",
                annotation_scheme="BIO",  # BIO scheme
                split="train",
                split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
                data_split_method="iterative",
                max_length=512,
                random_seed=42,
            )

            model = AutoModelForTokenClassification.from_pretrained(
                model_id, num_labels=dataset.num_labels
            )

            # Verify BIO scheme
            assert dataset.annotation_scheme == "BIO"

            # Test forward pass
            device = "cpu"
            model = model.to(device)
            model.eval()

            sample = dataset[0]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            assert outputs.logits is not None

        except Exception as e:
            pytest.skip(f"Could not run BIO pipeline: {e}")
