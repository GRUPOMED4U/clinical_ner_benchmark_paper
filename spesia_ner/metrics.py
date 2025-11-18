from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids.astype(np.float32)
    # logits: [batch, seq_len, num_labels]
    # labels: [batch, seq_len, num_labels]

    probs = 1 / (1 + np.exp(-logits))
    labels = labels.astype(int)

    # Flatten batch and sequence for metrics
    probs_flat = probs.reshape(-1, probs.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1])

    metrics = {}

    if probs.shape[-1] > 1:
        # Multilabel/multiclass case
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0
        best_threshold = 0.5
        for t in thresholds:
            preds_flat = (probs_flat > t).astype(int)
            f1 = f1_score(labels_flat, preds_flat, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        # Use best threshold for all metrics
        preds_flat = (probs_flat > best_threshold).astype(int)
        metrics['best_threshold'] = best_threshold
        metrics['macro_precision'] = precision_score(labels_flat, preds_flat, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(labels_flat, preds_flat, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(labels_flat, preds_flat, average='macro', zero_division=0)
        try:
            metrics['macro_auc'] = roc_auc_score(labels_flat, probs_flat, average='macro')
        except ValueError:
            metrics['macro_auc'] = 0.0

        metrics['micro_precision'] = precision_score(labels_flat, preds_flat, average='micro', zero_division=0)
        metrics['micro_recall'] = recall_score(labels_flat, preds_flat, average='micro', zero_division=0)
        metrics['micro_f1'] = f1_score(labels_flat, preds_flat, average='micro', zero_division=0)
        try:
            metrics['micro_auc'] = roc_auc_score(labels_flat, probs_flat, average='micro')
        except ValueError:
            metrics['micro_auc'] = 0.0
    else:
        # Single-label (binary) case
        probs_flat = probs_flat.reshape(-1)
        labels_flat = labels_flat.reshape(-1)
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0
        best_threshold = 0.5
        for t in thresholds:
            preds_flat = (probs_flat > t).astype(int)
            f1 = f1_score(labels_flat, preds_flat, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        # Use best threshold for all metrics
        preds_flat = (probs_flat > best_threshold).astype(int)
        metrics['best_threshold'] = best_threshold
        metrics['precision'] = precision_score(labels_flat, preds_flat, average='binary', zero_division=0)
        metrics['recall'] = recall_score(labels_flat, preds_flat, average='binary', zero_division=0)
        metrics['f1'] = f1_score(labels_flat, preds_flat, average='binary', zero_division=0)
        try:
            metrics['auc'] = roc_auc_score(labels_flat, probs_flat)
        except ValueError:
            metrics['auc'] = 0.0

    return metrics

def compute_metrics_with_per_label_thresholds(eval_pred, include_per_label_thresholds=False):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids.astype(np.float32)
    # logits: [batch, seq_len, num_labels]
    # labels: [batch, seq_len, num_labels]

    probs = 1 / (1 + np.exp(-logits))
    labels = labels.astype(int)

    # Flatten batch and sequence for metrics
    probs_flat = probs.reshape(-1, probs.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1])

    metrics = {}

    if probs.shape[-1] > 1:
        # Multilabel case — compute a threshold PER LABEL
        thresholds = np.linspace(0.01, 0.99, 99)
        n_labels = probs_flat.shape[-1]
        best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)

        for k in range(n_labels):
            y_true = labels_flat[:, k]
            p = probs_flat[:, k]
            best_f1_k = -1.0
            best_t_k = 0.5
            # Skip labels that are all one class to avoid degenerate optimization
            # (we still keep default 0.5)
            if (y_true.sum() == 0) or (y_true.sum() == y_true.shape[0]):
                best_thresholds[k] = best_t_k
                continue
            for t in thresholds:
                y_pred_k = (p > t).astype(int)
                f1_k = f1_score(y_true, y_pred_k, average='binary', zero_division=0)
                if f1_k > best_f1_k:
                    best_f1_k = f1_k
                    best_t_k = t
            best_thresholds[k] = best_t_k

        # Use per-label thresholds for final predictions
        preds_flat = (probs_flat > best_thresholds).astype(int)

        if include_per_label_thresholds:
            metrics['best_thresholds'] = best_thresholds

        metrics['best_thresholds_mean'] = best_thresholds.mean().item()

        metrics['macro_precision'] = precision_score(labels_flat, preds_flat, average='macro', zero_division=0)
        metrics['macro_recall']   = recall_score(labels_flat, preds_flat, average='macro', zero_division=0)
        metrics['macro_f1']       = f1_score(labels_flat, preds_flat, average='macro', zero_division=0)
        try:
            metrics['macro_auc']  = roc_auc_score(labels_flat, probs_flat, average='macro')
        except ValueError:
            metrics['macro_auc']  = 0.0

        metrics['micro_precision'] = precision_score(labels_flat, preds_flat, average='micro', zero_division=0)
        metrics['micro_recall']    = recall_score(labels_flat, preds_flat, average='micro', zero_division=0)
        metrics['micro_f1']        = f1_score(labels_flat, preds_flat, average='micro', zero_division=0)
        try:
            metrics['micro_auc']   = roc_auc_score(labels_flat, probs_flat, average='micro')
        except ValueError:
            metrics['micro_auc']   = 0.0

    else:
        # Single-label (binary) case — unchanged
        probs_flat = probs_flat.reshape(-1)
        labels_flat = labels_flat.reshape(-1)
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0.0
        best_threshold = 0.5
        for t in thresholds:
            preds_flat = (probs_flat > t).astype(int)
            f1 = f1_score(labels_flat, preds_flat, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        preds_flat = (probs_flat > best_threshold).astype(int)
        metrics['best_threshold'] = float(best_threshold)
        metrics['precision'] = precision_score(labels_flat, preds_flat, average='binary', zero_division=0)
        metrics['recall']    = recall_score(labels_flat, preds_flat, average='binary', zero_division=0)
        metrics['f1']        = f1_score(labels_flat, preds_flat, average='binary', zero_division=0)
        try:
            metrics['auc']   = roc_auc_score(labels_flat, probs_flat)
        except ValueError:
            metrics['auc']   = 0.0

    return metrics
