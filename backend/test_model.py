from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image


# ==============================
# RUTAS
# ==============================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "model"

MODEL_PATH = MODEL_DIR / "medpapaya_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
CLASS_INFO_PATH = MODEL_DIR / "class_info.json"

IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_first_image(dataset_dir: Path):
    """
    Busca automáticamente una imagen dentro del dataset para probar el modelo.
    """
    for class_folder in dataset_dir.iterdir():
        if class_folder.is_dir():
            for file in class_folder.iterdir():
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    return file
    return None


def prepare_image(image_path: Path):
    """
    Abre la imagen, la redimensiona y la convierte en arreglo NumPy.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def main():
    print("\n==============================================")
    print("        MEDPAPAYA - PRUEBA DEL MODELO")
    print("==============================================\n")

    if not MODEL_PATH.exists():
        print("ERROR: No se encontró el modelo entrenado.")
        print(MODEL_PATH)
        return

    if not CLASS_NAMES_PATH.exists():
        print("ERROR: No se encontró class_names.json.")
        print(CLASS_NAMES_PATH)
        return

    if not CLASS_INFO_PATH.exists():
        print("ERROR: No se encontró class_info.json.")
        print(CLASS_INFO_PATH)
        return

    print("Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH)

    class_names = load_json(CLASS_NAMES_PATH)
    class_info = load_json(CLASS_INFO_PATH)

    print("Modelo cargado correctamente.")

    test_image = find_first_image(DATASET_DIR)

    if test_image is None:
        print("ERROR: No se encontró ninguna imagen para probar.")
        return

    print("\nImagen usada para la prueba:")
    print(test_image)

    image_array = prepare_image(test_image)

    predictions = model.predict(image_array)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    info = class_info.get(predicted_class, {})
    label = info.get("label", predicted_class)
    recommendation = info.get("recommendation", "No hay recomendación registrada.")

    print("\n----------------------------------------------")
    print("RESULTADO DE LA PREDICCIÓN")
    print("----------------------------------------------")
    print(f"Clase detectada: {predicted_class}")
    print(f"Nombre visible: {label}")
    print(f"Confianza: {confidence * 100:.2f}%")
    print(f"Recomendación: {recommendation}")

    print("\n----------------------------------------------")
    print("PROBABILIDADES POR CLASE")
    print("----------------------------------------------")

    for class_name, probability in zip(class_names, predictions):
        visible_name = class_info.get(class_name, {}).get("label", class_name)
        print(f"{visible_name}: {probability * 100:.2f}%")

    print("\n==============================================")
    print("        PRUEBA FINALIZADA")
    print("==============================================\n")


if __name__ == "__main__":
    main()
