@echo off
setlocal
title MedPapaya - Lanzador
color 0A

echo =====================================================
echo              MEDPAPAYA - LANZADOR LOCAL
echo =====================================================
echo.

cd /d "%~dp0"

echo Carpeta del proyecto:
echo %cd%
echo.

echo Verificando estructura del proyecto...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: No se encontro el entorno virtual:
    echo .venv\Scripts\python.exe
    echo.
    echo Solucion:
    echo 1. Abre una terminal en esta carpeta.
    echo 2. Ejecuta: py -m venv .venv
    echo 3. Ejecuta: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b
)

if not exist "backend\main.py" (
    echo ERROR: No se encontro:
    echo backend\main.py
    echo.
    echo Este archivo BAT debe estar dentro de la carpeta principal:
    echo MEDPAPAYA online
    echo.
    pause
    exit /b
)

if not exist "backend\model\medpapaya_model.keras" (
    echo ERROR: No se encontro el modelo:
    echo backend\model\medpapaya_model.keras
    echo.
    pause
    exit /b
)

if not exist "frontend\index.html" (
    echo ERROR: No se encontro:
    echo frontend\index.html
    echo.
    pause
    exit /b
)

if not exist ".env" (
    echo ADVERTENCIA: No se encontro el archivo .env
    echo La pagina puede funcionar, pero Gemini podria fallar.
    echo.
)

echo Estructura correcta.
echo.

echo Cerrando servidores anteriores en el puerto 8000, si existen...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
    echo Cerrando proceso con PID %%a...
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo Iniciando servidor MedPapaya en una ventana aparte...
echo.

start "Servidor MedPapaya - NO CERRAR" /D "%~dp0" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo Esperando a que cargue TensorFlow, el modelo y la API...
echo Esto puede tardar varios segundos.
echo.

set ATTEMPT=0
set MAX_ATTEMPTS=60

:WAIT_SERVER
set /a ATTEMPT+=1

curl.exe -s --max-time 3 -f "http://127.0.0.1:8000/api/health" >nul 2>nul

if not errorlevel 1 (
    goto SERVER_READY
)

echo Esperando servidor... intento %ATTEMPT% de %MAX_ATTEMPTS%
timeout /t 3 >nul

if %ATTEMPT% GEQ %MAX_ATTEMPTS% (
    goto SERVER_ERROR
)

goto WAIT_SERVER

:SERVER_READY
echo.
echo =====================================================
echo              MEDPAPAYA ESTA LISTO
echo =====================================================
echo.
echo La API respondio correctamente en:
echo http://127.0.0.1:8000/api/health
echo.
echo Abriendo paginas para demostracion...
echo.

start "" "http://127.0.0.1:8000"
timeout /t 2 >nul

start "" "http://127.0.0.1:8000/docs"
timeout /t 2 >nul

start "" "http://127.0.0.1:8000/api/health"

echo.
echo Paginas abiertas:
echo.
echo 1. Pagina principal:
echo    http://127.0.0.1:8000
echo.
echo 2. Swagger Docs:
echo    http://127.0.0.1:8000/docs
echo.
echo 3. Health API:
echo    http://127.0.0.1:8000/api/health
echo.
echo En Swagger Docs puedes mostrar:
echo - GET /api/health
echo - POST /api/predict-one
echo - POST /api/predict
echo - POST /api/gemini-second-opinion
echo.
echo IMPORTANTE:
echo No cierres la ventana llamada:
echo Servidor MedPapaya - NO CERRAR
echo.
echo Para apagar MedPapaya, cierra esa ventana o presiona CTRL + C dentro de ella.
echo.
pause
exit /b

:SERVER_ERROR
echo.
echo =====================================================
echo                  ERROR AL INICIAR
echo =====================================================
echo.
echo El servidor no respondio despues de varios intentos.
echo.
echo Revisa la ventana llamada:
echo Servidor MedPapaya - NO CERRAR
echo.
echo Posibles causas:
echo - Error en backend\main.py
echo - Modelo no encontrado
echo - Dependencias faltantes
echo - Puerto 8000 ocupado
echo - Error en .env
echo - TensorFlow tardo demasiado en cargar
echo.
pause
exit /b