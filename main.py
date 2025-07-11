import os
import logging
import pandas as pd
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from datetime import datetime
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sql_connection.db import df_final

# ========== CONFIG ==========
PRIMARY_MODEL = "Salesforce/codet5-base"
BACKUP_MODEL = "google/flan-t5-xl"
LOG_DIR = "logs"
MODEL_DIR = "model"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== LOGGING ==========
log_file = os.path.join(LOG_DIR, "training_log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
def log(msg):
    print(msg)
    logging.info(msg)

log("🚀 Training pipeline started")
log(f"🖥️ CUDA Available: {torch.cuda.is_available()}")

# ========== LOAD DATA ==========
log("📥 Loading dataset")
try:
    df = df_final
    required_cols = ['Name', 'DisplayName', 'Description', 'Child_Relationship', 'Container', 'ObjectType', 'AppObjectConfiguration']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Required column '{col}' not found in DataFrame.")

    df_cleaned = df.copy()
    for col in required_cols:
        df_cleaned[col] = df_cleaned[col].astype(str).replace(r'^\s*$', pd.NA, regex=True).replace('nan', pd.NA)

    df_cleaned.dropna(subset=required_cols, inplace=True)
    if df_cleaned.empty:
        raise ValueError("❌ Dataset is empty after cleaning.")

    train_df, val_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

    def create_input_text(row):
        return f"Name: {row['Name']}; DisplayName: {row['DisplayName']}; Description: {row['Description']}; Child_Relationship: {row['Child_Relationship']}; ObjectType: {row['ObjectType']}; Configuration: {row['AppObjectConfiguration']}"

    train_df["input_text"] = train_df.apply(create_input_text, axis=1)
    val_df["input_text"] = val_df.apply(create_input_text, axis=1)

    train_dataset = Dataset.from_pandas(train_df[['input_text', 'Container']].rename(columns={"input_text": "description", "Container": "formio_json"}))
    val_dataset = Dataset.from_pandas(val_df[['input_text', 'Container']].rename(columns={"input_text": "description", "Container": "formio_json"}))

    log(f"✅ Dataset loaded: {len(train_dataset)} training rows, {len(val_dataset)} validation rows")

except Exception as e:
    log(f"❌ Failed to load dataset: {e}")
    raise

# ========== LOAD TOKENIZER & MODEL ==========
log("📦 Loading tokenizer and model")
def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        log(f"⚠️ Failed to load model '{model_name}': {e}")
        return None, None

tokenizer, model = load_model(PRIMARY_MODEL)
if tokenizer is None or model is None:
    log("🔁 Falling back to backup model...")
    tokenizer, model = load_model(BACKUP_MODEL)

if tokenizer is None or model is None:
    raise RuntimeError("❌ Could not load any model. Exiting.")

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# ========== PREPROCESS ==========
log("🧼 Tokenizing datasets")
def preprocess(example):
    input_text = example["description"] if pd.notna(example["description"]) else ""
    formio_json = example["formio_json"] if pd.notna(example["formio_json"]) else ""
    return tokenizer(
        text=input_text,
        text_target=formio_json,
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# ========== TRAINING ARGUMENTS ==========
log("⚙️ Setting training arguments")
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=10,
    report_to="none",
)

# ========== TRAINING ==========
log("🧠 Starting training")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
log("✅ Training complete")

# ========== EVALUATION ==========
log("📈 Starting evaluation")
predictions, references, structured_matches = [], [], []

for example in val_dataset:
    input_text = example["description"]
    reference_output = example["formio_json"]

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    output_tokens = model.generate(**inputs, max_length=512)
    decoded_pred = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    predictions.append(decoded_pred)
    references.append(reference_output)

    try:
        pred_json = json.loads(decoded_pred)
        ref_json = json.loads(reference_output)
        structured_matches.append(pred_json == ref_json)
    except Exception:
        structured_matches.append(False)

def compute_similarity(preds, refs):
    return sum([SequenceMatcher(None, p.strip(), r.strip()).ratio() for p, r in zip(preds, refs)]) / len(preds)

text_sim = compute_similarity(predictions, references)
struct_sim = sum(structured_matches) / len(structured_matches)

log(f"🎯 Text Similarity Score: {text_sim:.4f}")
log(f"🔍 Exact JSON Match Accuracy: {struct_sim:.4f}")

results_df = pd.DataFrame({
    "input": [ex["description"] for ex in val_dataset],
    "expected_json": references,
    "predicted_json": predictions,
    "exact_json_match": structured_matches
})
# results_df.to_csv(os.path.join(LOG_DIR, "predictions_vs_actual.csv"), index=False)
# log("📤 Saved predictions_vs_actual.csv")

# ========== SAVE MODEL ==========
log("💾 Saving model weights and tokenizer")
model_path = os.path.join(MODEL_DIR, "trained_model.pt")
torch.save(model.state_dict(), model_path)
tokenizer.save_pretrained(MODEL_DIR)
log(f"📦 Model and tokenizer saved to {MODEL_DIR}")
