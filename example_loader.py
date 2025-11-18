from pathlib import Path
from transformers import AutoTokenizer
from spesia_ner.datasets import (
    ClinicalRecordsDataset,
    DataCollatorForMultiLabelTokenClassification,
)
from torch.utils.data import DataLoader
import os


# ============================================================
# Caminhos de dados
# ============================================================

# Diretório base
BASE_DIR = Path("data")

# Subpastas específicas
PATHS = [
    BASE_DIR / "SemClinBr" / "annotated_records",
    BASE_DIR / "Spesia" / "argilla" / "annotated_records",
    BASE_DIR / "Spesia" / "doccano" / "annotated_records",
]

# Verifica se as pastas existem
for p in PATHS:
    if not p.exists():
        print(f"[AVISO] Diretório não encontrado: {p}")
    else:
        print(f"[OK] Diretório encontrado: {p}")


# ============================================================
# Configurações de tokenização
# ============================================================

TOKENIZER_MODEL = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


# ============================================================
# Carregamento unificado dos datasets
# ============================================================

all_records = []

for dataset_path in PATHS:
    if not dataset_path.exists():
        continue

    print(f"\nCarregando dataset em: {dataset_path}")

    dataset = ClinicalRecordsDataset(
        path=dataset_path,
        tokenizer=tokenizer,
        label_type="tags",  # 'tags' ou 'semantic_groups'
        split="train",
        split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
        annotation_scheme="BIO",
        min_samples_per_label=1,
    )

    print(f"→ {len(dataset.all_records)} registros carregados.")
    all_records.extend(dataset.all_records)

print(f"\n Total consolidado de registros: {len(all_records)}")


# ============================================================
# Exemplo prático de uso
# ============================================================

# Carrega um dos datasets (ex: Spesia Argilla)
dataset = ClinicalRecordsDataset(
    path=BASE_DIR / "Spesia" / "argilla" / "annotated_records",
    tokenizer=tokenizer,
    label_type="tags",
    split="train",
    annotation_scheme="BIO",
)

print(f"\nDataset Argilla → {len(dataset)} registros no split '{dataset.split}'")
print(f"Total de rótulos: {dataset.num_labels}")
print(f"Rótulos considerados: {dataset.labels_to_consider[:10]}")

# Exibe exemplo tokenizado
print("\nExemplo de registro tokenizado:")
dataset.print_record(0)


# ============================================================
# 5Criação de DataLoader
# ============================================================

collator = DataCollatorForMultiLabelTokenClassification(
    pad_token_id=tokenizer.pad_token_id, max_length=512, num_labels=dataset.num_labels
)

train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)

# Visualiza um batch
for batch in train_loader:
    print("\nBatch exemplo:")
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)
    break
