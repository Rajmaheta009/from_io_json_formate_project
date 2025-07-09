from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === SETUP ===
MODEL_NAME = "google/flan-t5-small"
MODEL_PATH = "model/from_io_json_formate.pth"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# === FASTAPI SETUP ===
app = FastAPI(title="MCP - Form.io Generator")

# Request Schema
class InputPayload(BaseModel):
    description: str

# Response Schema
class OutputPayload(BaseModel):
    formio_json: str

# === INFERENCE ROUTE ===
@app.post("/generate", response_model=OutputPayload)
def generate_formio_json(payload: InputPayload):
    try:
        inputs = tokenizer(payload.description, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print({"formio_json": generated})
        return {"formio_json": generated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
