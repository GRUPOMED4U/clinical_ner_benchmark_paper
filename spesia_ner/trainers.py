import torch
from transformers import Trainer, TrainerState, TrainingArguments, TrainerControl

from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 3):
        self.best_metric_less_is_better = float("inf")
        self.best_metric_greater_is_better = float("-inf")
        self.patience = patience
        self.num_steps_with_no_improvement = 0

    def on_epoch_end(self, 
                    args: TrainingArguments, 
                    state: TrainerState, 
                    control: TrainerControl, 
                    **kwargs):
        if state.best_metric is None: return control

        if args.greater_is_better:
            if state.best_metric > self.best_metric_greater_is_better:
                self.best_metric_greater_is_better = state.best_metric
                self.num_steps_with_no_improvement = 0
            else:
                self.num_steps_with_no_improvement += 1

        else:
            if state.best_metric < self.best_metric_less_is_better:
                self.best_metric_less_is_better = state.best_metric
                self.num_steps_with_no_improvement = 0
            else:
                self.num_steps_with_no_improvement += 1

        if self.num_steps_with_no_improvement >= self.patience:
            control.should_training_stop = True
        
        return control

class MultiLabelTokenTrainer(Trainer):
    def __init__(self, *args, pos_weight = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")                  # float tensor [B, L, C] with 0/1
        outputs = model(**inputs)
        logits = outputs.logits                        # [B, L, C]

        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(logits.device)              
             
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.pos_weight)
        loss = loss_fct(logits, labels.float())        # [B, L, C]
        # mask by attention
        attn = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        loss = (loss * attn).sum() / attn.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss