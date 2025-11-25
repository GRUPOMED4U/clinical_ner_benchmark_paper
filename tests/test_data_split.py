import sys
import types
from pathlib import Path

# Inject lightweight stubs for heavy optional dependencies so datasets import works
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    _torch.cuda = _Cuda()
    _torch.device = lambda x: x
    _torch.tensor = lambda *a, **k: None
    _torch.long = int
    _torch.stack = lambda x: x
    _torch.zeros = lambda *args, **kwargs: 0
    # provide torch.utils.data.Dataset so `from torch.utils.data import Dataset` works
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    data_mod.Dataset = Dataset
    sys.modules["torch"] = _torch
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoTokenizer = object
    _tf.AutoModelForTokenClassification = object
    token_utils = types.ModuleType("transformers.tokenization_utils_base")

    class BatchEncoding:
        pass

    token_utils.BatchEncoding = BatchEncoding
    _tf.tokenization_utils_base = token_utils
    sys.modules["transformers"] = _tf
    sys.modules["transformers.tokenization_utils_base"] = token_utils

if "skmultilearn" not in sys.modules:
    _sk = types.ModuleType("skmultilearn")
    _ms = types.ModuleType("skmultilearn.model_selection")

    def iterative_train_test_split(X, y, test_size=0.2):
        n = int(len(X) * (1 - test_size))
        X_temp = X[:n]
        y_temp = y[:n]
        X_test = X[n:]
        y_test = y[n:]
        return X_temp, y_temp, X_test, y_test

    _ms.iterative_train_test_split = iterative_train_test_split
    sys.modules["skmultilearn"] = _sk
    sys.modules["skmultilearn.model_selection"] = _ms

from spesia_ner import datasets


class FakeRecord:
    def __init__(self, text, tags):
        self.text = text
        self.tags = tags
        self.annotations = []  # not used by split logic


def test_same_text_not_split_across_partitions(tmp_path, monkeypatch):
    """
    Ensure multiple records with identical text are not split across different splits.
    The dataset should include ALL copies of a given text in the same split (or none).
    """
    # create dummy files
    dummy_files = []
    dummy_files += ["a.text" for _ in range(100)]
    dummy_files += ["b.text" for _ in range(100)]
    dummy_files += ["c.text" for _ in range(100)]

    for fname in dummy_files:
        (tmp_path / fname).write_text("dummy")

    # capture all created records by our fake loader
    created_records = []

    def fake_from_file(path):
        name = Path(path).name
        if "a" in name:
            rec = FakeRecord("A_TEXT", ["TAG_A"])
        elif "b" in name:
            rec = FakeRecord("B_TEXT", ["TAG_A"])
        elif "c" in name:
            rec = FakeRecord("C_TEXT", ["TAG_A"])
        else:
            rec = FakeRecord("DIFFERENT_TEXT", ["TAG_A"])

        created_records.append(rec)
        return [rec]

    # patch the loader used inside datasets module
    monkeypatch.setattr(datasets.Record, "from_file", fake_from_file)

    # instantiate dataset with low min_samples_per_label
    dataset = datasets.ClinicalRecordsDataset(
        path=tmp_path,
        tokenizer=None,
        min_samples_per_label=1,
        random_seed=0,
    )

    # counts in original created records
    total_counts = {}
    for r in created_records:
        total_counts[r.text] = total_counts.get(r.text, 0) + 1

    # counts in dataset.records (those included in the chosen split)
    split_counts = {}
    for r in dataset.records:
        split_counts[r.text] = split_counts.get(r.text, 0) + 1

    # For every text observed in original data, the count in the split must be either
    # 0 (text excluded from this split) or equal to the total count (all copies included).
    for text, tot in total_counts.items():
        in_split = split_counts.get(text, 0)
        assert in_split in (
            0,
            tot,
        ), f"Text '{text}' was partially split: {in_split}/{tot} in '{dataset.split}'"
