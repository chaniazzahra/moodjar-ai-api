from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
import torch
import os


app = FastAPI(title="MoodJar AI API")


MOOD_MODEL_PATH = "mood_model"
SUPPORT_MODEL_PATH = "support_model"

# Label sesuai dataset MoodJar kamu
ID2LABEL = {
    0: "bahagia",
    1: "cemas",
    2: "marah",
    3: "sedih"
}


class MoodRequest(BaseModel):
    text: str


class MoodResponse(BaseModel):
    modelName: dict
    text: str
    predictedLabel: str
    confidenceScore: float
    supportMessage: str


mood_tokenizer = None
mood_model = None
support_tokenizer = None
support_model = None


def load_models():
    global mood_tokenizer, mood_model, support_tokenizer, support_model

    if os.path.exists(MOOD_MODEL_PATH) and os.path.exists(os.path.join(MOOD_MODEL_PATH, "config.json")):
        mood_tokenizer = AutoTokenizer.from_pretrained(MOOD_MODEL_PATH)
        mood_model = AutoModelForSequenceClassification.from_pretrained(MOOD_MODEL_PATH)
        mood_model.eval()
        print("Mood classification model loaded.")
    else:
        print("Mood model belum ditemukan di folder mood_model.")

    if os.path.exists(SUPPORT_MODEL_PATH) and os.path.exists(os.path.join(SUPPORT_MODEL_PATH, "config.json")):
        support_tokenizer = AutoTokenizer.from_pretrained(SUPPORT_MODEL_PATH)
        support_model = AutoModelForSeq2SeqLM.from_pretrained(SUPPORT_MODEL_PATH)
        support_model.eval()
        print("Support message model loaded.")
    else:
        print("Support model belum ditemukan di folder support_model.")


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/")
def home():
    return {
        "message": "MoodJar AI API is running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "moodModelLoaded": mood_model is not None,
        "supportModelLoaded": support_model is not None
    }


def fallback_support_message(label: str) -> str:
    messages = {
        "bahagia": "Senang mendengar kamu sedang merasa bahagia. Nikmati momen baik ini dan tetap jaga energi positifmu ya.",
        "cemas": "Aku tahu rasa cemas bisa terasa berat. Coba tarik napas pelan-pelan, beri jeda sebentar, dan hadapi semuanya satu langkah dulu.",
        "marah": "Rasa marah itu wajar, tapi kamu tetap bisa mengendalikannya. Coba beri ruang untuk tenang sebelum merespons sesuatu.",
        "sedih": "Tidak apa-apa merasa sedih. Kamu tidak harus kuat setiap saat. Beri dirimu waktu untuk istirahat dan pulih perlahan."
    }
    return messages.get(label, "Terima kasih sudah berbagi perasaanmu. Semoga kamu bisa merasa lebih baik setelah ini.")


def predict_mood(text: str):
    if mood_model is None or mood_tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail="Mood model belum dimasukkan. Isi folder mood_model terlebih dahulu."
        )

    inputs = mood_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = mood_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, dim=1)

    label = ID2LABEL.get(predicted_id.item(), str(predicted_id.item()))
    confidence_score = round(confidence.item(), 4)

    return label, confidence_score


def generate_support_message(text: str, label: str):
    if support_model is None or support_tokenizer is None:
        return fallback_support_message(label)

    input_text = f"mood: {label} text: {text}"

    inputs = support_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        output_ids = support_model.generate(
            **inputs,
            max_length=80,
            num_beams=4,
            early_stopping=True
        )

    support_message = support_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    if not support_message.strip():
        return fallback_support_message(label)

    return support_message


@app.post("/predict", response_model=MoodResponse)
def predict(data: MoodRequest):
    if not data.text or not data.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text tidak boleh kosong."
        )

    label, confidence_score = predict_mood(data.text)
    support_message = generate_support_message(data.text, label)

    return {
        "modelName": {
            "classification": "indobenchmark/indobert-base-p1 fine-tuned MoodJar",
            "generation": "t5-small fine-tuned MoodJar"
        },
        "text": data.text,
        "predictedLabel": label,
        "confidenceScore": confidence_score,
        "supportMessage": support_message
    }