from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification
import json

# Load the training data
with open("data/processed/train_data.json", "r") as f:
    data = json.load(f)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Prepare the datasets
# Convert data into a format suitable for the model

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()