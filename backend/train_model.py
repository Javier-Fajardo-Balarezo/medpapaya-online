from pathlib import Path
import json
import tensorflow as tf


# ==========================================
# RUTAS DEL PROYECTO
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "model"

MODEL_PATH = MODEL_DIR / "medpapaya_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
CLASS_INFO_PATH = MODEL_DIR / "class_info.json"


# ==========================================
# CONFIGURACIÓN DEL ENTRENAMIENTO
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 123

EPOCHS_STAGE_1 = 25
EPOCHS_STAGE_2 = 10


# ==========================================
# INFORMACIÓN PARA MOSTRAR EN LA PÁGINA
# ==========================================

CLASS_INFO = {
    "Bacterial_Blight": {
        "label": "Tizón bacteriano",
        "recommendation": "Retirar hojas muy afectadas, evitar exceso de humedad y revisar la propagación en el cultivo."
    },
    "Carica_Insect_Hole": {
        "label": "Daño por insectos",
        "recommendation": "Revisar presencia de plagas, aplicar control preventivo y observar hojas jóvenes."
    },
    "healthy_leaf": {
        "label": "Hoja saludable",
        "recommendation": "No se observan signos claros de enfermedad. Mantener monitoreo y buenas prácticas agrícolas."
    },
    "Yellow_Necrotic_Spots_Holes": {
        "label": "Manchas necróticas amarillas",
        "recommendation": "Revisar hojas afectadas, controlar humedad y consultar con un especialista si las manchas aumentan."
    }
}


# ==========================================
# FUNCIÓN PARA CONTAR IMÁGENES POR CLASE
# ==========================================

def count_images_by_class(dataset_dir: Path):
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    counts = {}

    for class_folder in dataset_dir.iterdir():
        if class_folder.is_dir():
            images = [
                file for file in class_folder.iterdir()
                if file.is_file() and file.suffix.lower() in image_extensions
            ]
            counts[class_folder.name] = len(images)

    return counts


# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    print("\n==============================================")
    print("        MEDPAPAYA - ENTRENAMIENTO DEL MODELO")
    print("==============================================\n")

    if not DATASET_DIR.exists():
        print("ERROR: No se encontró la carpeta del dataset.")
        print(DATASET_DIR)
        return

    MODEL_DIR.mkdir(exist_ok=True)

    print("Dataset encontrado:")
    print(DATASET_DIR)

    image_counts = count_images_by_class(DATASET_DIR)

    print("\nCantidad de imágenes por clase:")
    for class_name, count in image_counts.items():
        print(f"- {class_name}: {count} imágenes")

    if len(image_counts) < 2:
        print("\nERROR: Se necesitan al menos 2 clases para entrenar.")
        return

    print("\nCargando imágenes para entrenamiento y validación...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("\nClases detectadas por TensorFlow:")
    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    print("\nCalculando pesos por clase para compensar desbalance...")

    total_images = sum(image_counts.values())
    class_weight = {}

    for index, class_name in enumerate(class_names):
        count = image_counts.get(class_name, 1)
        class_weight[index] = total_images / (num_classes * count)

    print("\nPesos por clase:")
    for index, weight in class_weight.items():
        print(f"{class_names[index]}: {weight:.3f}")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\nConstruyendo modelo EfficientNetB0...")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="data_augmentation")

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print("\n==============================================")
    print("ETAPA 1: Entrenando capas finales")
    print("==============================================\n")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_1,
        class_weight=class_weight,
        callbacks=callbacks
    )

    print("\n==============================================")
    print("ETAPA 2: Ajuste fino del modelo")
    print("==============================================\n")

    base_model.trainable = True

    # Congelamos la mayoría de capas y dejamos entrenar solo las últimas.
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Las capas BatchNormalization se dejan congeladas para evitar inestabilidad.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_2,
        class_weight=class_weight,
        callbacks=callbacks
    )

    print("\nEvaluando modelo final...")

    val_loss, val_accuracy = model.evaluate(val_ds)

    print("\n----------------------------------------------")
    print("RESULTADO FINAL")
    print("----------------------------------------------")
    print(f"Pérdida de validación: {val_loss:.4f}")
    print(f"Precisión de validación: {val_accuracy * 100:.2f}%")

    print("\nGuardando modelo y clases...")

    model.save(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    with open(CLASS_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_INFO, f, ensure_ascii=False, indent=4)

    print("\nArchivos guardados:")
    print(MODEL_PATH)
    print(CLASS_NAMES_PATH)
    print(CLASS_INFO_PATH)

    print("\n==============================================")
    print("        ENTRENAMIENTO FINALIZADO")
    print("==============================================\n")


if __name__ == "__main__":
    main()
