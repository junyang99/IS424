import pandas as pd
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset with error handling
def load_jsonl_subset(file_path, sample_size=30000):  # Changed to 30,000
    print(f"Loading dataset from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i + 1}: {line[:100]}")
                continue
    if not data:
        raise ValueError(f"No valid data found in {file_path}")
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df.sample(n=min(sample_size, len(df)), random_state=42)

# Paths
train_file_path = "train (1).jsonl"
dev_file_path = "shared_task_dev.jsonl"

# Load datasets
print("\nStarting dataset loading process...")
train_df = load_jsonl_subset(train_file_path, sample_size=30000)  # 30,000 Training Size
dev_df = load_jsonl_subset(dev_file_path, sample_size=5000)
print("✅ Datasets loaded successfully!")

# Convert labels
print("\nMapping labels to binary values...")
label_map = {"SUPPORTS": 1, "REFUTES": 0}
train_df = train_df[train_df['label'].isin(label_map)].copy()
dev_df = dev_df[dev_df['label'].isin(label_map)].copy()
train_df['label'] = train_df['label'].map(label_map)
dev_df['label'] = dev_df['label'].map(label_map)
print("✅ Label mapping complete!")

# Extract text and labels
print("\nExtracting texts and labels for tokenization...")
train_texts, train_labels = train_df['claim'].tolist(), train_df['label'].tolist()
dev_texts, dev_labels = dev_df['claim'].tolist(), dev_df['label'].tolist()
print("✅ Text extraction complete!")

# BERT Tokenization
print("\nInitializing BERT tokenizer...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

print("Tokenizing training dataset...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

print("Tokenizing validation dataset...")
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
print("✅ Tokenization complete!")

# Convert to Hugging Face Dataset
print("\nConverting tokenized data to Hugging Face Dataset format...")
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"].tolist(),
    "attention_mask": train_encodings["attention_mask"].tolist(),
    "labels": train_labels
})
dev_dataset = Dataset.from_dict({
    "input_ids": dev_encodings["input_ids"].tolist(),
    "attention_mask": dev_encodings["attention_mask"].tolist(),
    "labels": dev_labels
})
print("✅ Conversion complete!")

# Load BERT model
print("\nLoading BERT model...")
bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("✅ BERT model loaded successfully!")

# Training settings
print("\nConfiguring training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Save the model at every epoch instead of "no"
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,  # Log training updates every 50 steps
    save_total_limit=1,  # Only keep the best model to save disk space
    load_best_model_at_end=True
)
print("✅ Training arguments configured!")

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Train BERT model
print("\nStarting BERT model training...")
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
print("\n✅ BERT model training complete!")

# Evaluate Model
print("\nEvaluating BERT model...")
eval_results = trainer.evaluate()
print(f"📊 Final Evaluation Results: {eval_results}")

print("\n🚀 Script execution complete! BERT model is trained and evaluated successfully! ✅")
