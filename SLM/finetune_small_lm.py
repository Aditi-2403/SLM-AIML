# finetune_small_lm.py
"""
Fine-tune a small GPT-2 style model (distilgpt2 or tiny-gpt2).
Usage:
    python finetune_small_lm.py
Change MODEL_NAME below to:
    - "sshleifer/tiny-gpt2"  -> very small, very fast (bad quality, for testing)
    - "distilgpt2"           -> small, fast, decent quality
"""

import os
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "distilgpt2"      # 🔄 change to "sshleifer/tiny-gpt2" if you want tiny test model
OUTPUT_DIR = "./small_lm_ft"
TRAIN_FILE = "data/train.txt"  # make sure this file exists
VAL_FILE = "data/val.txt"      # make sure this file exists
BATCH_SIZE = 8
EPOCHS = 3
MAX_LENGTH = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Load tokenizer and model
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add [PAD] if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# ==============================
# Load dataset from text files
# ==============================
def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        texts = [l.strip() for l in f if l.strip()]
    return {"text": texts}

train = Dataset.from_dict(load_text_file(TRAIN_FILE))
val = Dataset.from_dict(load_text_file(VAL_FILE))
dataset = DatasetDict({"train": train, "validation": val})

# ==============================
# Tokenization
# ==============================
def tokenize_batch(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==============================
# Training setup
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    fp16=False,  # set to True if you have a modern GPU
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # keep only last 2 checkpoints
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

# ==============================
# Train
# ==============================
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Fine-tuning finished. Model saved to {OUTPUT_DIR}")
