from typing import List
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from .data_models import Annotation
from .datasets import (
    ClinicalRecordsDataset,
    DataCollatorForMultiLabelTokenClassification,
)


class AutoAnnotator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForTokenClassification,
        best_thresholds: list[float],
        idx_to_label: dict[int, str],
        max_length: int = 512,
        batch_size: int = 10,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.best_thresholds = best_thresholds
        self.idx_to_label = idx_to_label
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=self.max_length,
            num_labels=len(idx_to_label),
            device=self.device,
        )
        self.batch_size = batch_size

    def _predict(
        self,
        records: (
            transformers.tokenization_utils_base.BatchEncoding
            | list[transformers.tokenization_utils_base.BatchEncoding]
        ),
    ) -> np.ndarray:
        if isinstance(records, transformers.tokenization_utils_base.BatchEncoding):
            records = [records]

        output = self.model(**self.data_collator(records, include_labels=False))
        logits = output.logits.detach().cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        predictions = probs > self.best_thresholds
        del output, logits
        return predictions  # [batch_size, seq_len, num_labels]

    def _map_prediction_to_tokens(
        self,
        predictions: np.ndarray,  # [batch_size, seq_len, num_labels]
        records: List[transformers.tokenization_utils_base.BatchEncoding],
    ) -> list[list[Annotation]]:
        mapped_annotations = []

        for current_record, current_record_predictions in zip(records, predictions):
            annotations = []
            current_str_position = 0

            for token_index, pred in enumerate(current_record_predictions):
                text = self.tokenizer.decode(
                    current_record.input_ids[token_index], skip_special_tokens=True
                )
                if token_index >= len(current_record.input_ids):
                    break
                label_ids = np.where(pred == True)[0]
                start_pos = current_str_position
                end_pos = current_str_position + len(text)
                for label_id in label_ids:
                    annotations.append(
                        Annotation(
                            id=str(token_index),
                            tags=set([self.idx_to_label[label_id]]),
                            start=start_pos,
                            end=end_pos,
                            text=text,
                        )
                    )
                current_str_position = end_pos

            mapped_annotations.append(annotations)

        return mapped_annotations

    def _merge_annotations(self, annotations: list[Annotation]) -> list[Annotation]:
        annotations = deepcopy(annotations)
        # merge annotations
        merged_annotations = {}
        for ann in annotations:
            for tag in ann.tags:
                if (ann.start, tag) in merged_annotations:
                    old_ann = merged_annotations[(ann.start, tag)]
                    new_ann = Annotation(
                        id=old_ann.id,
                        tags=set([tag]),
                        start=old_ann.start,
                        end=ann.end,
                        text=old_ann.text + ann.text,
                    )
                    merged_annotations[(new_ann.end, tag)] = new_ann
                    del merged_annotations[(ann.start, tag)]
                else:
                    new_ann = Annotation(
                        id=ann.id,
                        tags=set([tag]),
                        start=ann.start,
                        end=ann.end,
                        text=ann.text,
                    )
                    merged_annotations[(new_ann.end, tag)] = new_ann

        return list(merged_annotations.values())

    def annotate(self, dataset_path: Path | str) -> ClinicalRecordsDataset:
        self.dataset = ClinicalRecordsDataset(dataset_path, self.tokenizer)

        # send model to GPU if available
        self.model.to(self.device)

        for idx in tqdm(
            range(0, len(self.dataset), self.batch_size),
            desc="Annotating records",
            total=(len(self.dataset)) // self.batch_size,
        ):
            batch = self.dataset[idx : idx + self.batch_size]
            predictions = self._predict(batch)
            annotations = self._map_prediction_to_tokens(predictions, batch)
            merged_annotations = [self._merge_annotations(ann) for ann in annotations]
            for record_idx in range(idx, min(idx + self.batch_size, len(self.dataset))):
                self.dataset.records[record_idx].annotations = merged_annotations[
                    record_idx - idx
                ]

        # send model back to CPU to free up memory and empty cache
        self.model.to("cpu")
        torch.cuda.empty_cache()

        return self.dataset
