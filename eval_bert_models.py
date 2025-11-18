import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path

from metrics import compute_metrics
from datasets import SemClinBrRecords, DataCollatorForMultiLabelTokenClassification
from trainers import MultiLabelTokenTrainer

models_to_evaluate = [
    "pucpr/biobertpt-all",
    # "pucpr/biobertpt-bio",
    # "pucpr/biobertpt-clin"
]

semantic_groups_to_consider = [
    'Activities & Behaviors',
    'Anatomy',
    'Chemicals & Drugs',
    'Concepts & Ideas',
    'Devices',
    'Disorders',
    'Genes & Molecular Sequences',
    'Geographic Areas',
    'Living Beings',
    'Objects',
    'Occupations',
    'Organizations',
    'Phenomena',
    'Physiology',
    'Procedures'
]

# Define relevant paths
plots_path = Path("plots")
plots_path.mkdir(exist_ok=True)

metrics_path = Path("metrics")
metrics_path.mkdir(exist_ok=True)

for model_id in models_to_evaluate:
    (plots_path/model_id.replace("/", "_")).mkdir(exist_ok=True)



for model_id in models_to_evaluate:
    all_metrics = []
    for semantic_group in semantic_groups_to_consider:
        print("Evaluating model: ", model_id)
        print("Semantic group: ", semantic_group)

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=len(semantic_groups_to_consider))

        train_dataset = SemClinBrRecords(
            'data/SemClinBr/annotated_records', 
            tokenizer, 
            semantic_groups_to_consider=[semantic_group],
            label_type='semantic_groups', 
            split="train")

        val_dataset = SemClinBrRecords(
            'data/SemClinBr/annotated_records', 
            tokenizer, 
            semantic_groups_to_consider=[semantic_group],
            label_type='semantic_groups', 
            split="val")

        test_dataset = SemClinBrRecords(
            'data/SemClinBr/annotated_records', 
            tokenizer, 
            semantic_groups_to_consider=[semantic_group],
            label_type='semantic_groups', 
            split="test")

        model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=train_dataset.num_labels)

        training_args = TrainingArguments(
            output_dir=Path("./output")/model_id.replace("/", "_")/semantic_group.replace(" ", "_").lower(),
            per_device_train_batch_size=50,
            num_train_epochs=10,
            learning_rate=2e-5,
            include_for_metrics=['inputs'],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=512,
            num_labels=train_dataset.num_labels
        )

        trainer = MultiLabelTokenTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        model_path = Path("models")/model_id.replace("/", "_")/semantic_group.replace(" ", "_").lower()
        model_path.mkdir(exist_ok=True, parents=True)
        trainer.save_model(model_path)

        pred_output = trainer.predict(test_dataset)

        metrics = pred_output.metrics
        metrics['model'] = model_id
        metrics['semantic_group'] = semantic_group

        all_metrics.append(metrics)
        pd.DataFrame.from_records(all_metrics).to_csv(metrics_path/f'{model_id.replace("/", "_")}.csv', index=False)

        # Get probabilities (sigmoid for multilabel)
        probs = 1 / (1 + np.exp(-pred_output.predictions))
        labels = pred_output.label_ids.astype(int)

        # For single-label: flatten arrays
        if probs.shape[-1] == 1:
            probs_flat = probs.reshape(-1)
            labels_flat = labels.reshape(-1)
        else:
            # For multilabel: plot for each label
            probs_flat = probs.reshape(-1, probs.shape[-1])
            labels_flat = labels.reshape(-1, labels.shape[-1])

        # Plot 
        for i in range(probs_flat.shape[-1] if probs_flat.ndim > 1 else 1):
            y_score = probs_flat
            y_true = labels_flat
            label_name = semantic_group

            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            plt.plot(recall, precision, label=label_name)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig(plots_path/model_id.replace("/", "_")/f"precision_recall_curve_{semantic_group.replace(' ', '_').lower()}.png")
        plt.clf()

        # Plot ROC curve 
        for i in range(probs_flat.shape[-1] if probs_flat.ndim > 1 else 1):
            y_score = probs_flat
            y_true = labels_flat
            label_name = semantic_group

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            plt.plot(fpr, tpr, label=f"{label_name} (AUC={auc:.2f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(plots_path/model_id.replace("/", "_")/f"roc_curve_{semantic_group.replace(' ', '_').lower()}.png")
        plt.clf()