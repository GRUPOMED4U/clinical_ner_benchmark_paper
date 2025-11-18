import matplotlib

matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import torch
import argparse
import json
import time
import multiprocessing
import matplotlib.pyplot as plt
import re  # <--- adicionado para limpeza de nomes

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.training_args import TrainingArguments as HFTrainingArguments
from transformers import EarlyStoppingCallback

from spesia_ner.datasets import (
    ClinicalRecordsDataset,
    DataCollatorForMultiLabelTokenClassification,
)
from spesia_ner.trainers import MultiLabelTokenTrainer
from spesia_ner.metrics import compute_metrics
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score


# =========================================================
# FUNÇÃO PARA EVITAR ERROS DE NOMES DE PASTAS NO WINDOWS
# =========================================================
def sanitize_filename(name: str) -> str:
    """Remove caracteres inválidos de nomes de diretórios/arquivos."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def main():
    # =========================================================
    # PARÂMETROS CLI
    # =========================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument(
        "--annotation_type", type=str, default="IO", choices=["IO", "BIO"]
    )
    args = parser.parse_args()

    # =========================================================
    # CONFIGURAÇÕES DE TREINAMENTO
    # =========================================================
    
    MODEL_ID = args.model_id
    ANNOT_TYPE = args.annotation_type

    DATASET_PATH = Path("data/Spesia/doccano/annotated_records")

    # Sanitiza o nome do modelo antes de criar diretório
    safe_model_name = sanitize_filename(MODEL_ID)
    OUTPUT_DIR = Path("results_fast") / f"{safe_model_name}_{ANNOT_TYPE}_doccano"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EPOCHS = 40
    LR = 5e-5
    PATIENCE = 10
    DELTA = 1e-4
    MAX_LEN = 512
    METRIC_FOR_BEST_MODEL = "eval_micro_f1" 

    LABELS_TO_IGNORE = []
    LABELS_TO_CONSIDER = []

    BATCH_SIZE, GRAD_ACCUM = 10, 5

    print(f"\n Iniciando benchmark: {MODEL_ID} ({ANNOT_TYPE})")
    print(f"Batch efetivo: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")

    # =========================================================
    # TOKENIZER E DATASETS
    # =========================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    def build_dataset(split):
        try:
            return ClinicalRecordsDataset(
                DATASET_PATH,
                tokenizer,
                split=split,
                label_type="tags",
                annotation_scheme=ANNOT_TYPE,
                tags_to_consider=LABELS_TO_CONSIDER,
                labels_to_ignore=LABELS_TO_IGNORE,
            )
        except TypeError:
            # fallback para versões antigas
            return ClinicalRecordsDataset(
                DATASET_PATH,
                tokenizer,
                split=split,
                label_type="tags",
                tags_to_consider=LABELS_TO_CONSIDER,
                labels_to_ignore=LABELS_TO_IGNORE,
            )

    print("Carregando datasets...")
    t0 = time.time()
    train_dataset = build_dataset("train")
    val_dataset = build_dataset("val")
    test_dataset = build_dataset("test")
    print(f"Datasets carregados em {time.time() - t0:.2f}s")

    # =========================================================
    # MODELO E PARÂMETROS DE TREINO
    # =========================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=train_dataset.num_labels
    ).to(device)

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    kwargs = dict(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",
        dataloader_num_workers=0,
    )

    # --- Compatibilidade entre versões do Transformers ---
    argnames = HFTrainingArguments.__init__.__code__.co_varnames
    if "evaluation_strategy" in argnames:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    training_args = HFTrainingArguments(**kwargs)

    data_collator = DataCollatorForMultiLabelTokenClassification(
        pad_token_id=tokenizer.pad_token_id,
        max_length=MAX_LEN,
        num_labels=train_dataset.num_labels,
    )

    # =========================================================
    # TREINAMENTO COM EARLY STOPPING
    # =========================================================
    print(f"Treinando em {device.upper()} | FP16: {use_fp16} | BF16: {use_bf16}")
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=PATIENCE,
    )

    trainer = MultiLabelTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        pos_weight=train_dataset.pos_weight,
    )

    trainer.train()

    # =========================================================
    # AVALIAÇÃO FINAL
    # =========================================================
    print("Avaliando no conjunto de teste...")
    trainer.eval_dataset = test_dataset
    results = trainer.evaluate()

    results.update(
        {
            "model_id": MODEL_ID,
            "annotation_type": ANNOT_TYPE,
            "batch_effective": BATCH_SIZE * GRAD_ACCUM,
            "patience": PATIENCE,
            "early_stopping_delta": DELTA,
        }
    )

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")

    # =========================================================
    # CURVAS (PR E ROC)
    # =========================================================
    print("Gerando curvas PR e ROC...")
    pred_output = trainer.predict(test_dataset)
    probs = 1 / (1 + np.exp(-pred_output.predictions))
    labels = pred_output.label_ids.astype(int)

    if probs.ndim == 3:
        probs_flat = probs.reshape(-1, probs.shape[-1])
        labels_flat = labels.reshape(-1, labels.shape[-1])
    else:
        probs_flat, labels_flat = probs.reshape(-1), labels.reshape(-1)

    # --- Precision–Recall ---
    plt.figure(figsize=(7, 6))
    for i in range(probs_flat.shape[-1] if probs_flat.ndim > 1 else 1):
        y_score = probs_flat[:, i] if probs_flat.ndim > 1 else probs_flat
        y_true = labels_flat[:, i] if probs_flat.ndim > 1 else labels_flat
        label_name = (
            test_dataset.labels_to_consider[i] if probs_flat.ndim > 1 else "Label"
        )
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=label_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({MODEL_ID} - {ANNOT_TYPE})")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()

    # --- ROC ---
    plt.figure(figsize=(7, 6))
    for i in range(probs_flat.shape[-1] if probs_flat.ndim > 1 else 1):
        y_score = probs_flat[:, i] if probs_flat.ndim > 1 else probs_flat
        y_true = labels_flat[:, i] if probs_flat.ndim > 1 else labels_flat
        label_name = (
            test_dataset.labels_to_consider[i] if probs_flat.ndim > 1 else "Label"
        )
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{label_name} (AUC={auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({MODEL_ID} - {ANNOT_TYPE})")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    print(f"\n Benchmark concluído com sucesso! Resultados em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    torch.manual_seed(42)
    np.random.seed(42)
    main()
