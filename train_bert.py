from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import torch

print("📥 Loading GoEmotions dataset...")
dataset = load_dataset("go_emotions", "simplified")

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
num_labels = dataset["train"].features["labels"].feature.num_classes
print(f"✅ Classes: {num_labels}")

def tokenize_and_encode(batch):
    enc = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    labels = []
    for labs in batch["labels"]:
        vec = [0] * num_labels
        for l in labs:
            vec[l] = 1
        labels.append(vec)
    enc["labels"] = labels
    return enc

encoded = dataset.map(tokenize_and_encode, batched=True)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, problem_type="multi_label_classification"
)

f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)  # threshold at 0 for BCEWithLogits
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_micro": f1.compute(predictions=preds, references=labels, average="micro")["f1"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

args = TrainingArguments(
    output_dir="bert_emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🚀 Training...")
trainer.train()
model.save_pretrained("./bert_emotion")
tokenizer.save_pretrained("./bert_emotion")
print("✅ Saved fine-tuned model to ./bert_emotion")
