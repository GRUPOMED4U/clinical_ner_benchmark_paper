# Clinical Named Entity Recognition in the Portuguese Language: A Benchmark of Modern BERT Models and LLMs

This repository contains the source code and experimental setup for the research paper "Clinical named entity recognition in the Portuguese language: a benchmark of modern BERT models and LLMs".

## 1. Abstract

Clinical notes hold valuable unstructured information, yet benchmarks for Named Entity Recognition (NER) in Portuguese remain scarce. This study evaluates BERT models and Large Language Models (LLMs) for clinical NER in Portuguese and tests strategies to address multilabel imbalance.

We compared BioBERTpt, BERTimbau, ModernBERT, and mmBERT with LLMs such as GPT-5 and Gemini-2.5, using the public SemClinBr corpus and a private breast-cancer dataset. Models were trained under identical conditions and evaluated with precision, recall, and F1-scores.

Key Findings:
- mmBERT-base achieved the best results (micro $F1=0.76$), outperforming other models in the SemClinBr.
- Both BioBERTpt and mmBERT performed well on the private breast cancer dataset.
- Iterative stratification improved class balance and overall performance.
- Multilingual BERT models perform strongly for Portuguese clinical NER and offer the advantage of running locally with limited resources when compared to LLMs.
- Data contamination cannot be excluded in the case of mmBERT models.

## 2. How to use this repository

### Prerequisites

Before setting up the environment, ensure you have the following installed:

- **Python** `>= 3.12`
- **Git**
- **`uv`** for dependency management. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- (Optional) **CUDA** compatible with the PyTorch wheels if you plan to use a GPU

All project dependencies are declared in `pyproject.toml` and their resolved versions are available in `uv.lock`.

### Installation and environment setup

This section explains how to prepare a local environment (Windows PowerShell) to run the repository scripts and models using `uv`.

#### Step 1: Clone the repository

```powershell
git clone <repo-url>
cd .\clinical_ner_benchmark_paper\clinical_ner_benchmark_paper
```

#### Step 2: Sync dependencies with `uv`

```powershell
uv sync
```

This command creates a virtual environment and installs all dependencies from `pyproject.toml` and `uv.lock`.

#### Step 3: Install PyTorch (choose according to your machine)

**Example (CUDA 12.8 — adjust according to your CUDA version):**

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

**Example (CPU-only):**

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

See https://pytorch.org/get-started/locally/ for the correct command for your setup.


**Running the tests**

- **Scripts:** Use `./scripts/run_tests.ps1` (Windows PowerShell) or `./scripts/run_tests.sh` (Linux/macOS) to create a virtual environment, install a minimal set of test dependencies and run `pytest`.
- **Activate `venv`:** PowerShell: `& .\.venv\Scripts\Activate.ps1` then run `pytest -q` to execute the test suite interactively.
- **Smoke tests (optional):** These tests require heavy ML libraries (PyTorch, `transformers`). Install them before running smoke tests. Example CPU-only install: `uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0` and `uv pip install transformers`. Run `pytest tests/test_model_smoke.py` to run only the smoke tests.
- **Without `uv` (alternative):** Create a venv manually: `python -m venv .venv; & .\.venv\Scripts\Activate.ps1; python -m pip install -U pip setuptools wheel` then install `pytest` and any other dependencies you need.
- **CI:** A GitHub Actions workflow is provided at `.github/workflows/ci-tests.yml` and runs the minimal test set on push/pull requests.

### Running scripts and notebooks

After environment setup, run scripts and notebooks using `uv run`:

```powershell
uv run python benchmark/run_benchmark.py
```

### Quick example: Using the bundled fake dataset

To learn how to use the `spesia_ner` package classes and functions, open and run the example notebook:

```
examples/dummy_data_example.ipynb
```

This notebook demonstrates:
- Loading the bundled dummy dataset (`data/dummy_data/fake_data.jsonl`)
- Creating `ClinicalRecordsDataset` objects with different annotation schemes (IO and BIO)
- Splitting data into train/val/test sets using iterative stratification
- Integrating HuggingFace tokenizers
- Accessing tokenized samples and computing label statistics
- Exporting datasets to different formats

All operations in this notebook use the fake data, making it fully reproducible without requiring access to private clinical datasets.

The repository includes filename sanitization to avoid common Windows path issues.

### Accessing datasets

#### SemClinBr dataset

Request access to the SemClinBr dataset following the official documentation [here](https://github.com/HAILab-PUCPR/SemClinBr).

#### Breast cancer dataset

This dataset was not made publicly available for privacy reasons.

## 3. Repository structure

```
|-- /benchmark               # Scripts for reproducing the results reported in the manuscript
|-- /data                    # Dummy data is provided as a quick start
|-- /examples                # Multiple jupyter notebooks are provided as an example of how to use the spesia_ner library
|-- /predictions             # Includes generations from LLMs tested
|-- /results                 # Inlucdes plots and metrics for various models tested
|-- /scripts                 # Includes various scripts for running automated tests
|-- /spesia_ner
|       |-- autolabeling.py  # For dataset autolabeling with trained model
|       |-- data_models.py   # Includes data models used
|       |-- datasets.py      # Dataset base class
|       |-- metrics.py       # Metrics used
|       |-- trainers.py      # Custom trainers used
|-- /tests                   # Automated tests for sanity check
|-- prompts.yaml             # Prompts used for LLM generations
```