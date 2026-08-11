from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://medpapaya-online.onrender.com/"

# CAMBIA ESTA RUTA POR LA RUTA REAL DE UNA IMAGEN DE PAPAYA
IMAGEN = Path(r"C:\Users\joseb\Downloads\papaya_enferma.png")

if not IMAGEN.exists():
    raise FileNotFoundError(f"No se encontró la imagen: {IMAGEN}")

driver = webdriver.Edge()
wait = WebDriverWait(driver, 120)

try:
    print("1. Abriendo MedPapaya...")
    driver.maximize_window()
    driver.get(URL)

    print("2. Buscando el selector de imágenes...")
    selector = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'input[type="file"]')
        )
    )

    print("3. Cargando imagen...")
    selector.send_keys(str(IMAGEN.resolve()))

    print("4. Buscando botón Analizar imágenes...")
    boton = wait.until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//button[contains(normalize-space(.),'Analizar imágenes')]"
            )
        )
    )

    print("5. Ejecutando análisis...")
    boton.click()

    print("6. Esperando diagnóstico...")

    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//*[contains(normalize-space(.),'Confianza:')]"
            )
        )
    )

    print("")
    print("========================================")
    print("PRUEBA SELENIUM: APROBADA")
    print("MedPapaya generó un diagnóstico.")
    print("========================================")

    driver.save_screenshot("resultado_selenium.png")

    input("Presiona ENTER para cerrar el navegador...")

finally:
    driver.quit()