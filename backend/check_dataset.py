from pathlib import Path
from PIL import Image
import pandas as pd


# Ruta base:
# C:\Users\joseb\Downloads\MEDPAPAYA\backend
PROJECT_DIR = Path(__file__).resolve().parent

# Ruta del dataset:
# C:\Users\joseb\Downloads\MEDPAPAYA\backend\dataset
DATASET_DIR = PROJECT_DIR / "dataset"

# Ruta del CSV:
# C:\Users\joseb\Downloads\MEDPAPAYA\backend\dataset\papaya_soil_dataset.csv
CSV_PATH = DATASET_DIR / "papaya_soil_dataset.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def check_image(image_path: Path) -> bool:
    """
    Verifica si una imagen puede abrirse correctamente.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def read_csv_safely(csv_path: Path):
    """
    Lee el CSV usando coma como separador principal.
    Si falla, intenta con punto y coma.
    """
    try:
        return pd.read_csv(csv_path)
    except Exception:
        try:
            return pd.read_csv(csv_path, sep=";")
        except Exception as e:
            print("ERROR: No se pudo leer el CSV.")
            print(e)
            return None


def main():
    print("\n==============================================")
    print("        MEDPAPAYA - REVISION DEL DATASET")
    print("==============================================\n")

    print("Ruta del backend:")
    print(PROJECT_DIR)

    print("\nRuta del dataset:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        print("\nERROR: No se encontro la carpeta 'dataset'.")
        print("Verifica que exista esta ruta:")
        print(DATASET_DIR)
        return

    print("\nDataset encontrado correctamente.")

    class_folders = [
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]

    if not class_folders:
        print("\nERROR: No se encontraron carpetas de clases dentro de 'dataset'.")
        return

    print("\nClases encontradas:\n")

    total_images = 0
    corrupted_images = []

    for class_folder in class_folders:
        images = [
            file for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        total_images += len(images)

        print(f"- {class_folder.name}: {len(images)} imagenes")

        for image_path in images:
            if not check_image(image_path):
                corrupted_images.append(str(image_path))

    print("\n----------------------------------------------")
    print("RESUMEN GENERAL DEL DATASET")
    print("----------------------------------------------")
    print(f"Total de clases: {len(class_folders)}")
    print(f"Total de imagenes: {total_images}")
    print(f"Imagenes danadas: {len(corrupted_images)}")

    if corrupted_images:
        print("\nImagenes danadas encontradas:")
        for img in corrupted_images:
            print(f"- {img}")
    else:
        print("\nNo se encontraron imagenes danadas.")

    print("\n----------------------------------------------")
    print("REVISION DEL CSV")
    print("----------------------------------------------")

    if not CSV_PATH.exists():
        print("\nNo se encontro el archivo CSV:")
        print(CSV_PATH)
        print("\nEsto no impide entrenar el modelo con imagenes.")
        return

    print("\nCSV encontrado:")
    print(CSV_PATH)

    df = read_csv_safely(CSV_PATH)

    if df is None:
        return

    print("\nColumnas encontradas en el CSV:")
    for col in df.columns:
        print(f"- {col}")

    print(f"\nCantidad de filas en el CSV: {len(df)}")

    expected_columns = [
        "image_id",
        "disease_label",
        "soil_ph",
        "nitrogen",
        "phosphorus",
        "potassium",
        "moisture",
        "temperature",
        "soil_health"
    ]

    missing_columns = [
        col for col in expected_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nColumnas esperadas que no se encontraron:")
        for col in missing_columns:
            print(f"- {col}")
    else:
        print("\nEl CSV tiene todas las columnas esperadas.")

    if "disease_label" in df.columns:
        print("\nRegistros por clase segun el CSV:\n")
        print(df["disease_label"].value_counts())

    print("\n----------------------------------------------")
    print("COMPARACION ENTRE CSV E IMAGENES")
    print("----------------------------------------------")

    if "image_id" not in df.columns:
        print("\nNo se puede comparar porque el CSV no tiene la columna 'image_id'.")
        return

    dataset_images = set()

    for class_folder in class_folders:
        for file in class_folder.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                dataset_images.add(file.name)

    csv_images = set(df["image_id"].astype(str).tolist())

    missing_in_folders = csv_images - dataset_images
    missing_in_csv = dataset_images - csv_images

    print(f"\nImagenes mencionadas en el CSV pero no encontradas en carpetas: {len(missing_in_folders)}")
    print(f"Imagenes encontradas en carpetas pero no mencionadas en el CSV: {len(missing_in_csv)}")

    if missing_in_folders:
        print("\nEjemplos de imagenes que estan en el CSV pero no en carpetas:")
        for img in list(missing_in_folders)[:10]:
            print(f"- {img}")

    if missing_in_csv:
        print("\nEjemplos de imagenes que estan en carpetas pero no en el CSV:")
        for img in list(missing_in_csv)[:10]:
            print(f"- {img}")

    print("\n==============================================")
    print("        REVISION FINALIZADA CORRECTAMENTE")
    print("==============================================\n")


if __name__ == "__main__":
    main()
