import os
import json
from io import BytesIO
from pathlib import Path
from typing import List

from dotenv import load_dotenv

import numpy as np
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from google import genai
from google.genai import types


# =====================================================
# RUTAS DEL PROYECTO
# =====================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(PROJECT_DIR / ".env")

FRONTEND_DIR = PROJECT_DIR / "frontend"
ASSETS_DIR = PROJECT_DIR / "assets"

MODEL_DIR_OPTIONS = [
    BACKEND_DIR / "model",
    PROJECT_DIR / "model"
]

MODEL_DIR = None

for option in MODEL_DIR_OPTIONS:
    if option.exists():
        MODEL_DIR = option
        break

if MODEL_DIR is None:
    raise FileNotFoundError(
        "No se encontró la carpeta del modelo. "
        "Debe existir backend/model o model."
    )

MODEL_PATH = MODEL_DIR / "medpapaya_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
CLASS_INFO_PATH = MODEL_DIR / "class_info.json"

IMG_SIZE = (224, 224)
MAX_FILES = 50
LOW_CONFIDENCE_THRESHOLD = 51.0


# =====================================================
# CREACIÓN DE LA API
# =====================================================

app = FastAPI(
    title="MedPapaya API",
    description="API para clasificación de enfermedades foliares en papaya.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# CARGA DEL MODELO
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo entrenado: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró class_names.json: {CLASS_NAMES_PATH}"
    )

if not CLASS_INFO_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró class_info.json: {CLASS_INFO_PATH}"
    )


print("==========================================")
print("MEDPAPAYA API - CARGANDO MODELO")
print("==========================================")
print("Modelo:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

with open(CLASS_INFO_PATH, "r", encoding="utf-8") as file:
    class_info = json.load(file)

print("Modelo cargado correctamente.")
print("Clases disponibles:")

for class_name in class_names:
    print("-", class_name)

print("API lista para recibir imágenes.")


# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def prepare_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo procesar la imagen: {str(error)}"
        )


def predict_image(image_bytes: bytes, filename: str) -> dict:
    image_array = prepare_image(image_bytes)

    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    info = class_info.get(predicted_class, {})

    label = info.get("label", predicted_class)
    recommendation = info.get(
        "recommendation",
        "No se encontró una recomendación registrada para esta clase."
    )

    probability_list = []

    for index, probability in enumerate(probabilities):
        technical_class = class_names[index]
        info_item = class_info.get(technical_class, {})

        probability_list.append({
            "class_name": technical_class,
            "label": info_item.get("label", technical_class),
            "probability": round(float(probability * 100), 2)
        })

    probability_list = sorted(
        probability_list,
        key=lambda item: item["probability"],
        reverse=True
    )

    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

    return {
        "filename": filename,
        "prediction": {
            "class_name": predicted_class,
            "label": label,
            "confidence": round(confidence, 2),
            "recommendation": recommendation,
            "probabilities": probability_list
        },
        "low_confidence": low_confidence,
        "suggest_gemini": low_confidence,
        "message": (
            "El modelo tiene baja certeza. Se recomienda solicitar una segunda opinión con Gemini."
            if low_confidence
            else "El modelo generó una predicción con confianza suficiente."
        )
    }


# =====================================================
# ENDPOINTS PRINCIPALES
# =====================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "MedPapaya API funcionando correctamente.",
        "model_loaded": True,
        "classes": class_names
    }


@app.post("/api/predict-one")
async def predict_one(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo enviado debe ser una imagen."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="La imagen enviada está vacía."
        )

    return predict_image(image_bytes, file.filename)


@app.post("/api/predict")
async def predict_many(files: List[UploadFile] = File(...)):
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar al menos una imagen."
        )

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Solo se permite analizar hasta {MAX_FILES} imágenes."
        )

    results = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "El archivo no es una imagen válida."
            })
            continue

        image_bytes = await file.read()

        if not image_bytes:
            results.append({
                "filename": file.filename,
                "error": "La imagen está vacía."
            })
            continue

        results.append(predict_image(image_bytes, file.filename))

    return {
        "total": len(results),
        "results": results
    }


# =====================================================
# ENDPOINT GEMINI - SEGUNDA OPINIÓN
# =====================================================

@app.post("/api/gemini-second-opinion")
async def gemini_second_opinion(file: UploadFile = File(...)):
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se configuró GEMINI_API_KEY en el servidor. "
                "Agrega la clave como variable de entorno."
            )
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo enviado debe ser una imagen."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="La imagen enviada está vacía."
        )

    try:
        client = genai.Client(api_key=gemini_api_key)

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=file.content_type
        )

        prompt = """
Actúa como un asistente agropecuario especializado en observación visual de hojas de papaya.

Analiza la imagen enviada y responde en español de forma breve, clara y directa.

No uses Markdown.
No uses asteriscos.
No uses negritas.
No uses símbolos como **, ###, -, viñetas largas o tablas.
No des explicaciones extensas.
No repitas ideas.
No afirmes un diagnóstico definitivo.

Responde exactamente con esta estructura:

1. Observación visual:
Describe en 2 o 3 líneas lo más importante que se observa en la hoja.

2. Posible problema:
Indica la causa más probable según la imagen. Puede ser hongo, bacteria, virus, plaga, estrés ambiental, daño mecánico o deficiencia nutricional.

3. Nivel de seguridad:
Responde solo con una palabra: Bajo, Medio o Alto.

4. Recomendación:
Da 3 acciones concretas y prácticas para el usuario.

5. Advertencia:
Indica en una sola línea que esta es una segunda opinión orientativa y no reemplaza a un especialista agrícola.
"""

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

        response = client.models.generate_content(
            model=gemini_model,
            contents=[
                image_part,
                prompt
            ]
        )

        return {
            "filename": file.filename,
            "source": "Gemini API",
            "model": gemini_model,
            "analysis": response.text
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al consultar Gemini API: {str(error)}"
        )


# =====================================================
# SERVIR ARCHIVOS ESTÁTICOS
# =====================================================

if ASSETS_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=ASSETS_DIR),
        name="assets"
    )

if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend"
    )