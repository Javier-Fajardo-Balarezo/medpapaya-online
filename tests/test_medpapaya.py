from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app, prepare_image, IMG_SIZE


client = TestClient(app)


# =========================================================
# PRUEBAS UNITARIAS
# =========================================================

def test_prepare_image_correctamente():
    """
    Comprueba de forma aislada que prepare_image()
    transforme una imagen al formato requerido por el modelo.
    """

    imagen = Image.new("RGB", (100, 100), color="green")

    buffer = BytesIO()
    imagen.save(buffer, format="JPEG")

    resultado = prepare_image(buffer.getvalue())

    assert isinstance(resultado, np.ndarray)
    assert resultado.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)
    assert resultado.dtype == np.float32


def test_prepare_image_invalida():
    """
    Comprueba que prepare_image() rechace datos
    que no representan una imagen válida.
    """

    try:
        prepare_image(b"esto no es una imagen")
        assert False

    except HTTPException as error:
        assert error.status_code == 400


# =========================================================
# PRUEBAS DE INTEGRACIÓN
# =========================================================

def test_health_api():
    """
    Comprueba la integración FastAPI + configuración
    del modelo mediante el endpoint health.
    """

    respuesta = client.get("/api/health")

    assert respuesta.status_code == 200

    datos = respuesta.json()

    assert datos["status"] == "ok"
    assert datos["model_loaded"] is True
    assert len(datos["classes"]) > 0


def test_predict_one_integracion():
    """
    Comprueba la integración completa:
    API -> archivo -> preprocesamiento -> modelo -> respuesta.
    """

    imagen = Image.new("RGB", (224, 224), color="green")

    buffer = BytesIO()
    imagen.save(buffer, format="JPEG")
    buffer.seek(0)

    respuesta = client.post(
        "/api/predict-one",
        files={
            "file": (
                "prueba.jpg",
                buffer.getvalue(),
                "image/jpeg"
            )
        }
    )

    assert respuesta.status_code == 200

    datos = respuesta.json()

    assert "filename" in datos
    assert "prediction" in datos
    assert "class_name" in datos["prediction"]
    assert "confidence" in datos["prediction"]
    assert "probabilities" in datos["prediction"]