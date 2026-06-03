import json
import pickle
import re
import numpy as np
import tensorflow as tf
import tf_keras as keras

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
from tf_keras.models import load_model

app = FastAPI(
    title="MoodJar AI API",
    description="API prediksi mood dan generate support message tanpa ChatGPT/Gemini API",
    version="1.0.0"
)

MODEL_REPO = "chaniaa09/moodjar-ai-model"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename="saved_cnn/model.keras")
tok_path = hf_hub_download(repo_id=MODEL_REPO, filename="saved_cnn/tok.pkl")
le_path = hf_hub_download(repo_id=MODEL_REPO, filename="saved_cnn/le.pkl")
cfg_path = hf_hub_download(repo_id=MODEL_REPO, filename="saved_cnn/cfg.pkl")

model = load_model(model_path)

with open(tok_path, "rb") as f:
    tokenizer = pickle.load(f)

with open(le_path, "rb") as f:
    le = pickle.load(f)

with open(cfg_path, "rb") as f:
    config = pickle.load(f)

t5_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    subfolder="saved_t5_model"
)

t5_model = TFAutoModelForSeq2SeqLM.from_pretrained(
    MODEL_REPO,
    subfolder="saved_t5_model"
)


class MoodRequest(BaseModel):
    text: str


def predict_mood(text):
    seq = tokenizer.texts_to_sequences([text.lower()])

    pad = keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=config["maxlen"],
        padding="post",
        truncating="post"
    )

    probs = model.predict(pad, verbose=0)[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    cnn_label = le.inverse_transform([pred_class])[0]

    daily_activity_words = [
        "menunggu bus", "menunggu kereta", "menunggu angkot",
        "makan", "minum", "tidur", "mandi", "belajar",
        "kuliah", "rapat", "bekerja", "berangkat kerja", "pulang kerja"
    ]

    emotional_words = [
        "senang", "bahagia", "gembira",
        "sedih", "kecewa", "menangis",
        "marah", "kesal", "jengkel",
        "cemas", "khawatir", "takut",
        "panik", "gelisah",
        "stres", "stress", "capek"
    ]

    text_lower = text.lower()

    contains_daily_activity = any(w in text_lower for w in daily_activity_words)

    contains_emotional_word = any(
        re.search(rf"\b{re.escape(w)}\b", text_lower)
        for w in emotional_words
    )

    if confidence < 0.45 and not contains_emotional_word:
        final_label = "biasa saja"
    elif contains_daily_activity and not contains_emotional_word:
        final_label = "biasa saja"
    else:
        final_label = cnn_label

    all_confidences = {
        le.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(probs))
    }

    return {
        "predictedLabel": final_label,
        "confidenceScore": confidence
    }


def generate_support(text, predicted_label):
    input_text = f"mood: {predicted_label} | text: {text}"

    inputs = t5_tokenizer(
        input_text,
        return_tensors="tf",
        truncation=True
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
        "model_repo": MODEL_REPO,
        "classification": "CNN",
        "generation": "Fine-tuned T5",
        "external_ai_api": False
    }


@app.post("/predict")
def predict(request: MoodRequest):
    mood_result = predict_mood(request.text)

    support_message = generate_support(
        request.text,
        mood_result["predictedLabel"]
    )

    return {
        "text": request.text,
        "predictedLabel": mood_result["predictedLabel"],
        "confidenceScore": mood_result["confidenceScore"],
        "supportMessage": support_message,
        "modelName": {
            "classification": "CNN",
            "generation": "Fine-tuned T5"
        }
    }