import os
import logging
import pandas as pd
import torch
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
from sql_connection.db import df_final

# ========== SETUP ==========
MODEL_NAME = "google/flan-t5-base"
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

log("🚀 Training started")

# ========== LOAD DATA ==========
log("📥 Loading dataset")
try:
    df = df_final
    required_cols = ['Name', 'DisplayName', 'Description', 'Child_Relationship', 'Container']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Required column '{col}' not found in CSV.")

    df_cleaned = df.copy()
    for col in required_cols:
        df_cleaned[col] = df_cleaned[col].astype(str).replace(r'^\s*$', pd.NA, regex=True).replace('nan', pd.NA)

    df_cleaned.dropna(subset=required_cols, inplace=True)
    if df_cleaned.empty:
        raise ValueError("❌ Dataset is empty after cleaning.")

    # Combine input fields into one input string
    df_cleaned["input_text"] = df_cleaned.apply(
        lambda
            row: f"Name: {row['Name']}; DisplayName: {row['DisplayName']}; Description: {row['Description']}; Child_Relationship: {row['Child_Relationship']}; ObjectType: {row['ObjectType']}; Configuration: {row['AppObjectConfiguration']}",
        axis=1
    )

    dataset = Dataset.from_pandas(df_cleaned[['input_text', 'Container']].rename(
        columns={"input_text": "description", "Container": "formio_json"}
    ))

    log(f"✅ Dataset loaded with {len(dataset)} rows")

except Exception as e:
    log(f"❌ Failed to load dataset: {e}")
    raise

# ========== LOAD TOKENIZER & MODEL ==========
log("📦 Loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ========== PREPROCESS ==========
log("🧼 Tokenizing dataset")
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

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ========== TRAINING ARGUMENTS ==========
log("⚙️ Setting training arguments")

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="no",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=10,
    report_to="none",
)

# # ========== TRAINING ==========
# log("🧠 Starting training")
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     tokenizer=tokenizer,
#     data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
# )
#
# trainer.train()
# log("✅ Training complete")
#
# # ========== EVALUATION ==========
# log("📈 Starting evaluation")
#
# predictions, references = [], []
# for example in dataset:
#     input_text = example["description"]
#     reference_output = example["formio_json"]
#
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     output_tokens = model.generate(**inputs, max_length=512)
#     decoded_pred = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
#
#     predictions.append(decoded_pred)
#     references.append(reference_output)
#
# def compute_similarity(preds, refs):
#     return sum([SequenceMatcher(None, p.strip(), r.strip()).ratio() for p, r in zip(preds, refs)]) / len(preds)
#
# similarity = compute_similarity(predictions, references)
# log(f"🎯 Similarity Score: {similarity:.4f}")
#
# results_df = pd.DataFrame({
#     "input": [ex["description"] for ex in dataset],
#     "expected_json": references,
#     "predicted_json": predictions
# })
# results_df.to_csv(os.path.join(LOG_DIR, "predictions_vs_actual.csv"), index=False)
# log("📤 Saved predictions_vs_actual.csv")
#
#
# # ========== SAVE MODEL ==========
# log("💾 Saving model weights as .pth")
# import torch
#
# model_path = os.path.join(MODEL_DIR, "flan_t5_small_model.pt")
# torch.save(model.state_dict(), model_path)
# tokenizer.save_pretrained(MODEL_DIR)
# log(f"📦 Model weights saved to {model_path}")
