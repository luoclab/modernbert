from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


# Dataset id from huggingface.co/dataset
dataset_id = "argilla/synthetic-domain-text-classification"

# Load raw dataset
train_dataset = load_dataset("modernbert/synthetic-domain-text-classification", split='train')

split_dataset = train_dataset.train_test_split(test_size=0.1)
split_dataset['train'][5:7]
# if is_wandb_available():
import wandb
wandb.login(key="37163ecd07dc01f6a37f45037fb17c921eaf86f7")
# import transformers

from transformers import AutoTokenizer

# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("modernbert/modernbert_base")

# Tokenize helper function
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True,padding=True, return_tensors="pt")

# Tokenize dataset
if "label" in split_dataset["train"].features.keys():
    split_dataset =  split_dataset.rename_column("label", "labels") # to match Trainer
tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])

from transformers import AutoModelForSequenceClassification

# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base" 

# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    "modernbert/modernbert_base", num_labels=num_labels, label2id=label2id, id2label=id2label,
)
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# model_id = "answerdotai/ModernBERT-base"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForMaskedLM.from_pretrained("modernbert/modernbert_base")

from transformers import Trainer, TrainingArguments

# Define training args
training_args = TrainingArguments(
    output_dir= "ModernBERT-domain-classifier",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=1,
    bf16=True, # bfloat16 training
    optim="adamw_torch_fused", # improved optimizer
    # logging & evaluation strategies
    logging_strategy="steps",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)
trainer.train()