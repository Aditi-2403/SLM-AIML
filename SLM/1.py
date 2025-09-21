from transformers import TrainingArguments
from dataclasses import fields

# Check if evaluation_strategy exists
print([f.name for f in fields(TrainingArguments) if f.name == "evaluation_strategy"])
