import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFAutoModelForSeq2SeqLM
)

app = FastAPI(
    title="MoodJar AI API",
    description="API prediksi mood dan generate support message tanpa ChatGPT/Gemini API",
    version="1.0.0"
)

INDOBERT_PATH = "saved_indobert"
T5_PATH = "saved_t5_model"
ID2LABEL_PATH = "id2label.json"

with open(ID2LABEL_PATH, "r") as f:
    id2label = json.load(f)

indobert_tokenizer = AutoTokenizer.from_pretrained(INDOBERT_PATH)
indobert_model = TFAutoModelForSequenceClassification.from_pretrained(
    INDOBERT_PATH
)

t5_tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
t5_model = TFAutoModelForSeq2SeqLM.from_pretrained(T5_PATH)


class MoodRequest(BaseModel):
    text: str


def predict_mood(text: str):
    inputs = indobert_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    outputs = indobert_model(**inputs)
    logits = outputs.logits

    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))

    predicted_label = id2label[str(pred_id)]

    return predicted_label, confidence


def generate_support_message(text: str, predicted_label: str):
    input_text = f"mood: {predicted_label} | text: {text}"

    inputs = t5_tokenizer(
        input_text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=256
    )

    output = t5_model.generate(
        **inputs,
        max_new_tokens=300,
        min_new_tokens=25,
        num_beams=4,
        repetition_penalty=2.0,
        early_stopping=True
    )

    support_message = t5_tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return support_message


@app.get("/")
def root():
    return {
        "message": "MoodJar AI API is running",
        "model": "Local IndoBERT + Local T5",
        "external_ai_api": False
    }


@app.post("/predict")
def predict(request: MoodRequest):
    predicted_label, confidence_score = predict_mood(request.text)

    support_message = generate_support_message(
        request.text,
        predicted_label
    )

    return {
        "text": request.text,
        "predictedLabel": predicted_label,
        "confidenceScore": confidence_score,
        "supportMessage": support_message
    }