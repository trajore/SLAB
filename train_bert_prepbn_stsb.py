import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
from datasets import load_dataset
import evaluate
from functools import partial
import numpy as np

# Import LinearNorm and RepBN from your repo
from classification.models.prepbn import LinearNorm, RepBN

# Patch BERT to use LinearNorm instead of LayerNorm
class BertWithLinearNorm(BertForSequenceClassification):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._replace_layernorm_with_linearnorm()

    def _replace_layernorm_with_linearnorm(self):
        ln = partial(nn.LayerNorm, eps=1e-6)
        def convert(module):
            for name, child in module.named_children():
                if isinstance(child, nn.LayerNorm):
                    linearnorm = LinearNorm(child.normalized_shape[0], norm1=ln, norm2=RepBN, step=60000)
                    setattr(module, name, linearnorm)
                else:
                    convert(child)
        convert(self)


# Load dataset and metric
dataset = load_dataset('glue', 'stsb')
metric = evaluate.load('glue', 'stsb')

# Tokenizer and preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

dataset = dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.squeeze(predictions)
    return metric.compute(predictions=predictions, references=labels)

# Model and training args
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=1)
model = BertWithLinearNorm(config)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',  # Save checkpoint every epoch
    save_total_limit=3,     # Keep only last 3 checkpoints (optional)
    logging_dir='./logs',   # Directory for logs
    logging_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='pearson',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
)

if __name__ == '__main__':
    trainer.train()
    trainer.evaluate()
    # Save the final model
    trainer.save_model('./results/final_model')
