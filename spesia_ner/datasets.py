from pathlib import Path
import jsonlines
import pandas as pd
from typing import List, Set, Union, Literal, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
import random

from .data_models import Record

 

# ==============================================================
# Utility functions
# ==============================================================


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    """Check strict overlap of half-open intervals [start, end)."""
    return (a_start < b_end) and (b_start < a_end)


# ==============================================================
# Main dataset class
# ==============================================================


class ClinicalRecordsDataset(Dataset):
    """
    Dataset class for annotated clinical records (SemClinBr, Argilla, Docanno).
    Loads records from different sources and prepares data for multilabel NER model training.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: AutoTokenizer | None = None,
        label_type: Literal["tags", "semantic_groups"] = "tags",
        tags_to_consider: list[str] = [],
        semantic_groups_to_consider: list[str] = [],
        labels_to_ignore: list[str] = [],
        max_length=512,
        random_seed=42,
        split: Literal['train', 'val', 'test'] | None = None,
        split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
        min_samples_per_label: int = 10,
        semantic_group_standard: Literal["UMLS"] = "UMLS",
        annotation_scheme: Literal["IO", "BIO"] = "IO",
        begin_token_weight_scaler: float = 1.0,
        data_split_method: Literal["random", "iterative"] = "iterative",
    ) -> None:

        if isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path

        assert label_type in [
            "tags",
            "semantic_groups",
        ], "label_type must be 'tags' or 'semantic_groups'"

        assert annotation_scheme in [
            "IO",
            "BIO",
        ], "annotation_scheme must be 'IO' or 'BIO'"

        # --------------------------
        # Main attributes
        # --------------------------
        self.label_type = label_type
        self.records: list[Record] = []
        self.all_records: list[Record] = []
        self.split = split
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.min_samples_per_label = min_samples_per_label
        self.semantic_group_standard = semantic_group_standard
        self.tags: Set[str] = set()
        self.semantic_groups: List[str] = []
        self.labels_to_ignore: List[str] = labels_to_ignore
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.annotation_scheme = annotation_scheme
        self.begin_token_weight_scaler = begin_token_weight_scaler
        self.data_split_method = data_split_method

        # --------------------------
        # Load files
        # --------------------------
        self._load_all_records()

        # --------------------------
        # Get labels
        # --------------------------
        self._get_records_tags()

        if self.label_type == "semantic_groups":
            self._get_semantic_groups()
            self._map_tags_to_semantic_groups(standard=self.semantic_group_standard)

        # --------------------------
        # Define labels
        # --------------------------
        self.tags_to_consider = self.tags if not tags_to_consider else tags_to_consider
        self.tags_to_consider = [
            t for t in self.tags_to_consider if t not in self.labels_to_ignore
        ]

        # if self.label_type == "tags":
        #     assert len(self.tags_to_consider) > 0, "No valid tags found."

        self.semantic_groups_to_consider = (
            self.semantic_groups
            if not semantic_groups_to_consider
            else semantic_groups_to_consider
        )
        self.semantic_groups_to_consider = [
            t
            for t in self.semantic_groups_to_consider
            if t not in self.labels_to_ignore
        ]

        # if self.label_type == "semantic_groups":
        #     assert (
        #         len(self.semantic_groups_to_consider) > 0
        #     ), "No valid semantic groups found."

        # --------------------------
        # Create label map
        # --------------------------
        self._create_label_map()
        self.num_labels = len(self.labels_to_consider)

        # --------------------------
        # Split data
        # --------------------------
        if self.split is not None:
            self._split_records()
        else:
            self.records = self.all_records
            self.all_records = []

    # ==============================================================
    # Main methods
    # ==============================================================

    def _load_all_records(self) -> None:
        """
        Load all records from different sources. Handles different file formats. Raises error file format is not supported.
        """
        file_paths = list(self.path.glob("*"))
        if not file_paths:
            raise FileNotFoundError(f"No files found in {self.path}")

        for file_path in tqdm(file_paths, desc="Loading all Records"):
            try:
                records = Record.from_file(file_path)
                self.all_records.extend(records)
            except Exception as e:
                print(f"[WARN] Erro ao processar {file_path.name}: {e}")

    def _split_records(self) -> None:
        if self.data_split_method == "random":
            self._random_data_split()
        elif self.data_split_method == "iterative":
            self._iterative_stratification()
    
    def _random_data_split(self) -> None:
        random.seed(self.random_seed) 
        random.shuffle(self.all_records)

        # Compute split indices
        n_total = len(self.all_records)
        n_train = int(n_total * self.split_ratio["train"])
        n_val = int(n_total * self.split_ratio["val"])

        split_records = {
            "train": self.all_records[:n_train], 
            "val": self.all_records[n_train:n_train + n_val], 
            "test": self.all_records[n_train + n_val:]
            }

        self.records = split_records[self.split]
    
    def _iterative_stratification(self) -> None:
        np.random.seed(self.random_seed)

        # The same record may have more than one set of annotations
        ## Index records by unique text
        labels = getattr(self, self.label_type)
        unique_texts = {r.text: [0 for _ in labels] for r in self.all_records}

        ## Map annotations to index
        for r in self.all_records:
            for label in labels:
                if label in getattr(r, self.label_type):
                    unique_texts[r.text][labels.index(label)] = 1

        ## Create X and y
        X = np.array(list(unique_texts.keys())).reshape(-1, 1)
        y = np.array(list(unique_texts.values()))

        ## Remove labels with low sample count
        col_sums = y.sum(axis=0)
        counts = pd.Series(col_sums, index=labels)
        labels_to_ignore = counts[counts < self.min_samples_per_label].index.tolist()
        labels_to_keep = counts[counts >= self.min_samples_per_label].index.tolist()
        self.labels_to_ignore = labels_to_ignore
        print("Ignored labels during data split due to low sample count:", labels_to_ignore)

        ## Select only frequent labels
        keep_idx = [getattr(self, self.label_type).index(l) for l in labels_to_keep]
        y_filtered = y[:, keep_idx]

        ## Stratified split
        X_temp, y_temp, X_test, y_test = iterative_train_test_split(
            X, y_filtered, test_size=self.split_ratio["test"]
        )
        val_prop = self.split_ratio["val"] / (
            self.split_ratio["train"] + self.split_ratio["val"]
        )
        X_train, y_train, X_val, y_val = iterative_train_test_split(
            X_temp, y_temp, test_size=val_prop
        )

        records_split = {
            "train": X_train.reshape(-1).tolist(),
            "val": X_val.reshape(-1).tolist(),
            "test": X_test.reshape(-1).tolist(),
        }

        ## Define records to include in this split
        records_to_include_in_split = set(records_split[self.split])
        self.records = [r for r in self.all_records if r.text in records_to_include_in_split]

        ## Save memory
        self.all_records = []


    def adaptive_oversample(self, target_ratio: float = 0.8, random_state=42, max_replication: int = 15):
        """
        Adaptive oversampling using roulette wheel selection for individual sampling.
        Prioritizes records with multiple minority labels through probabilistic selection.
        
        Args:
            target_ratio: Percentage of the average to be reached (0.8 = 80%)
            random_state: Seed for reproducibility
            max_replication: Maximum number of replications per record
        """

        if self.split != 'train':
            return
        
        original_state = random.getstate() if random_state is not None else None
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        label_count_dict = self.get_label_count(self.label_type).to_dict()
        total_ocurrencias = sum(label_count_dict.values())
        num_labels = len(label_count_dict)
        avg_ocurrencias = total_ocurrencias / num_labels
        target_count = int(avg_ocurrencias * target_ratio)
              
        #Identify which labels need oversampling
        labels_needing_oversample = {}
        for label, count in label_count_dict.items():
            if count < target_count:
                deficit = target_count - count
                replication_factor = max(1, min(max_replication, deficit // (count + 1)))
                labels_needing_oversample[label] = {
                    'current_count': count,
                    'target_count': target_count,
                    'deficit': deficit,
                    'replication_factor': replication_factor
                }
        
        #WHEEL ROULETTE
        minority_labels_set = set(labels_needing_oversample.keys())
        
        high_value_records = []
        medium_value_records = []
        low_value_records = []
        
        for record in self.records:
            record_labels = getattr(record, self.label_type)
            minority_labels_in_record = [label for label in record_labels if label in minority_labels_set]
            
            if not minority_labels_in_record:
                continue
                
            num_minority_labels = len(minority_labels_in_record)
            
            if num_minority_labels >= 3:
                high_value_records.append({
                    'record': record,
                    'minority_labels': minority_labels_in_record,
                    'fitness': 3.0
                })
            elif num_minority_labels == 2:
                other_labels = [label for label in record_labels if label not in minority_labels_set]
                
                if len(other_labels) >= 3:
                    medium_value_records.append({
                        'record': record,
                        'minority_labels': minority_labels_in_record,
                        'fitness': 1.5  
                    })
                else:
                    low_value_records.append({
                        'record': record,
                        'minority_labels': minority_labels_in_record,
                        'fitness': 1.0  
                    })
        
        all_candidates = high_value_records + medium_value_records + low_value_records
        
        if not all_candidates:
            print("⚠️ There is no samples of minorities classes for oversampling")
            return
        
        def wheel_roulette_selection(candidates, num_selections):
            if not candidates:
                return []
            
            total_fitness = sum(candidate['fitness'] for candidate in candidates)
            
            wheel = []
            cumulative = 0.0
            
            for candidate in candidates:
                probability = candidate['fitness'] / total_fitness
                cumulative += probability
                wheel.append((cumulative, candidate))
            
            selected = []
            for _ in range(num_selections):
                spin = random.random()
                
                for threshold, candidate in wheel:
                    if spin <= threshold:
                        selected.append(candidate)
                        break
            
            return selected
        
        replication_needs = {}
        for label, info in labels_needing_oversample.items():
            replication_needs[label] = info['deficit']
        
        new_records = self.records.copy()
        replication_stats = {label: 0 for label in labels_needing_oversample.keys()}
        remaining_deficits = {label: info['deficit'] for label, info in labels_needing_oversample.items()}
        
        max_rounds = 15
        for round_num in range(max_rounds):
            if all(deficit <= 0 for deficit in remaining_deficits.values()):
                break
                
            total_remaining_deficit = sum(remaining_deficits.values())
            if total_remaining_deficit <= 0:
                break
                
            max_replicas_per_round = min(total_remaining_deficit, len(all_candidates) * 2)
            
            selected_candidates = wheel_roulette_selection(all_candidates, max_replicas_per_round)
            
            for candidate in selected_candidates:
                record = candidate['record']
                minority_labels = candidate['minority_labels']
                
                needed_for_any_label = any(
                    remaining_deficits.get(label, 0) > 0 
                    for label in minority_labels
                )
                
                if needed_for_any_label:
                    new_records.append(record)
                    
                    for label in minority_labels:
                        if label in replication_stats and remaining_deficits.get(label, 0) > 0:
                            replication_stats[label] += 1
                            remaining_deficits[label] = max(0, remaining_deficits[label] - 1)
            
        remaining_significant_deficits = {
            label: deficit for label, deficit in remaining_deficits.items() 
            if deficit > len(all_candidates) // 2  
        }
        
        if remaining_significant_deficits:
            for label, deficit in remaining_significant_deficits.items():
                label_candidates = [
                    candidate for candidate in all_candidates 
                    if label in candidate['minority_labels']
                ]
                
                if label_candidates:
                    selected_for_label = wheel_roulette_selection(label_candidates, deficit)
                    
                    for candidate in selected_for_label:
                        new_records.append(candidate['record'])
                        replication_stats[label] += 1
                        remaining_deficits[label] -= 1
                        
                        if remaining_deficits[label] <= 0:
                            break
        
        if random_state is not None:
            random.shuffle(new_records)
        
        self.records = new_records
        
        
    # ==============================================================
    # Label getters and mapping methods
    # ==============================================================

    @property
    def pos_weight(self):
        # Class weights for weighted loss if needed
        label_count = self.get_label_count(self.label_type)
        pos_weight = []
        for label, idx in self.label_map.items():
            base_label = label.split("-")[-1]
            curr_label_count = label_count.get(base_label, 0)

            # evita divisão por zero
            if curr_label_count == 0:
                curr_weight = 1.0
            else:
                curr_weight = (len(self) - curr_label_count) / max(curr_label_count, 1)

            if label.startswith("B-"):
                curr_weight *= self.begin_token_weight_scaler

            pos_weight.append(curr_weight)

        return torch.tensor(pos_weight)

    def _get_records_tags(self) -> None:
        tags = set()
        for record in self.all_records:
            tags.update(record.tags)
        self.tags = sorted(tags)

    def _get_semantic_groups(self, standard: str = "UMLS") -> None:
        if standard == "UMLS":
            semgroups_path = Path("data/SemClinBr/SemGroups.txt")
            semgroups = pd.read_csv(semgroups_path, sep="|", header=None)
            self.semantic_groups.extend(semgroups[1])
            self.semantic_groups = sorted(set(self.semantic_groups))
        else:
            raise NotImplementedError(
                f"Semantic group mapping to {standard} not implemented."
            )

    def _map_tags_to_semantic_groups(self, standard: str = "UMLS") -> None:
        if standard == "UMLS":
            semgroups_path = Path("data/SemClinBr/SemGroups.txt")
            semgroups = pd.read_csv(semgroups_path, sep="|", header=None)
            tag_to_semgroup = dict(zip(semgroups[3], semgroups[1]))

            for record in tqdm(
                self.all_records, desc="Mapping tags to UMLS semantic groups"
            ):
                for annotation in record.annotations:
                    annotation.semantic_groups = list(
                        {
                            tag_to_semgroup[tag]
                            for tag in annotation.tags
                            if tag in tag_to_semgroup
                        }
                    )
        else:
            raise NotImplementedError(
                f"Semantic group mapping to {standard} not implemented."
            )

    # ==============================================================
    # Tokenização e geração de labels
    # ==============================================================

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> transformers.tokenization_utils_base.BatchEncoding | List[transformers.tokenization_utils_base.BatchEncoding]:
        if isinstance(index, int):
            record = self.records[index]
            encoded_text, token_vectors = self._spans_to_multilabels(record)
            labels = torch.tensor(token_vectors, dtype=torch.float)
            encoded_text["labels"] = labels
            del encoded_text["offset_mapping"]
            return encoded_text
        
        elif isinstance(index, slice):
            encoded_texts = []
            records = self.records[index]
            for record in records:
                encoded_text, token_vectors = self._spans_to_multilabels(record)
                labels = torch.tensor(token_vectors, dtype=torch.float)
                encoded_text["labels"] = labels
                del encoded_text["offset_mapping"]
                encoded_texts.append(encoded_text)
            return encoded_texts

        else:
            raise ValueError("Index must be an integer or a slice.")

    def _create_label_map(self) -> None:
        self.labels_to_consider = getattr(self, f"{self.label_type}_to_consider")

        if self.annotation_scheme == "BIO":
            extended_labels = []
            for label in self.labels_to_consider:
                extended_labels += [f"B-{label}", f"I-{label}"]
            self.labels_to_consider = extended_labels

        self.label_map = {tag: i for i, tag in enumerate(self.labels_to_consider)}
        self.idx_to_label = {i: tag for i, tag in enumerate(self.labels_to_consider)}

    def _spans_to_multilabels(
        self, record: Record
    ) -> Tuple[transformers.tokenization_utils_base.BatchEncoding, List[List[int]]]:
        encoded_text = self.tokenizer(
            record.text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        offsets = encoded_text["offset_mapping"]
        word_ids = encoded_text.word_ids()
        token_labels = [set() for _ in offsets]

        if self.annotation_scheme == "IO":
            for annotation in record.annotations:
                labels_source = (
                    annotation.tags
                    if self.label_type == "tags"
                    else annotation.semantic_groups
                )
                for label in labels_source:
                    if label not in getattr(self, f"{self.label_type}_to_consider"):
                        continue
                    for i, (token_start, token_end) in enumerate(offsets):
                        if token_start == token_end:
                            continue
                        if overlaps(
                            token_start, token_end, annotation.start, annotation.end
                        ):
                            token_labels[i].add(label)

        elif self.annotation_scheme == "BIO":
            for annotation in record.annotations:
                labels_source = (
                    annotation.tags
                    if self.label_type == "tags"
                    else annotation.semantic_groups
                )
                for label in labels_source:
                    if label not in getattr(self, f"{self.label_type}_to_consider"):
                        continue
                    for i, (token_start, token_end) in enumerate(offsets):
                        if token_start == token_end:
                            continue
                        if overlaps(
                            token_start, token_end, annotation.start, annotation.end
                        ):
                            tag = (
                                f"B-{label}"
                                if token_start == annotation.start
                                else f"I-{label}"
                            )
                            token_labels[i].add(tag)

        token_vectors = []
        for i, word_id in enumerate(word_ids):
            token_vector = [0] * self.num_labels
            if word_id is not None:
                for label in token_labels[i]:
                    token_vector[self.label_map[label]] = 1
            token_vectors.append(token_vector)

        return encoded_text, token_vectors

    # ==============================================================
    # Statistics and visualization
    # ==============================================================

    def get_label_count(
        self, label_type: Literal["tags", "semantic_groups"] = "tags"
    ) -> pd.Series:
        labels = getattr(self, label_type)
        y_arr = np.array(
            [
                [1 if label in getattr(r, label_type) else 0 for label in labels]
                for r in self.records
            ],
            dtype=int,
        )
        col_sums = y_arr.sum(axis=0)
        return pd.Series(col_sums, index=labels).sort_values(ascending=False)

    def print_record(self, index: int):
        record = self[index]
        for token, labels in zip(record.input_ids, record.labels):
            curr_labels = (labels == 1).nonzero().view(-1)
            if len(curr_labels) == 0:
                labels_text = "O"
            else:
                labels_text = "|".join(
                    [self.idx_to_label[i.item()] for i in curr_labels]
                )
            print(f"{self.tokenizer.decode(token):20} - {labels_text}")

    # ==============================================================
    #  Data export methods
    # ==============================================================

    def export(self, 
               path: Path | str, 
               format: Literal["jsonl"],
               include_annotations: bool = False,
               exclude_duplicates: bool = True,
               ) -> None:
        """
        Exports the record to a file in the specified format and path.
        """
        if isinstance(path, str):
            path = Path(path)

        records_to_export = self.records
        
        if format == "jsonl":
            records_in_jsonl_format = []
            seen = set()

            for r in records_to_export:
                if r.text == '': continue

                r_to_export = {}
                r_to_export["text"] = r.text
                r_to_export["label"] = []
                
                if include_annotations:
                    labels = []
                    for ann in r.annotations:
                        for label in getattr(ann, self.label_type):
                            labels.append([ann.start, ann.end, label])
                    r_to_export["label"] = labels

                if exclude_duplicates and r_to_export["text"] in seen:
                    continue

                seen.add(r_to_export["text"])
                records_in_jsonl_format.append(r_to_export)

            with jsonlines.open(path, mode='w') as writer:
                for obj in records_in_jsonl_format:
                    writer.write(obj)


# ==============================================================
# Data Collator
# ==============================================================

class DataCollatorForMultiLabelTokenClassification:
    """
    Custom data collator for consistent padding and batching.
    """

    def __init__(self, pad_token_id=0, max_length=512, num_labels=1, device="cpu"):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.num_labels = num_labels
        self.device = device

    def __call__(self, features, include_labels=True):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))

            if include_labels:
                if labels.shape[0] < self.max_length:
                    pad_labels = torch.zeros(
                        self.max_length - labels.shape[0], self.num_labels
                    )
                    labels = torch.cat([labels, pad_labels], dim=0)
                elif labels.shape[0] > self.max_length:
                    labels = labels[: self.max_length]

                batch_labels.append(labels)

        if include_labels:
            return {
                "input_ids": torch.stack(batch_input_ids).to(self.device),
                "attention_mask": torch.stack(batch_attention_mask).to(self.device),
                "labels": torch.stack(batch_labels).to(self.device),
            }
        
        return {
                "input_ids": torch.stack(batch_input_ids).to(self.device),
                "attention_mask": torch.stack(batch_attention_mask).to(self.device),
            }
