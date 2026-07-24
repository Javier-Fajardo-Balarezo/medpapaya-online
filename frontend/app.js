const API_URL = "/api/predict-one";
const GEMINI_API_URL = "/api/gemini-second-opinion";

const MAX_FILES = 50;
const LOW_CONFIDENCE_LIMIT = 51;

const imageInput = document.getElementById("imageInput");
const fileCounter = document.getElementById("fileCounter");
const previewContainer = document.getElementById("previewContainer");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const resultsContainer = document.getElementById("resultsContainer");
const loadingBox = document.getElementById("loadingBox");
const historyContainer = document.getElementById("historyContainer");

const geminiImageInput = document.getElementById("geminiImageInput");
const geminiFileCounter = document.getElementById("geminiFileCounter");
const geminiPreviewContainer = document.getElementById("geminiPreviewContainer");
const geminiAnalyzeBtn = document.getElementById("geminiAnalyzeBtn");
const geminiClearBtn = document.getElementById("geminiClearBtn");
const geminiLoadingBox = document.getElementById("geminiLoadingBox");
const geminiStandaloneResult = document.getElementById("geminiStandaloneResult");

let selectedFiles = [];
let selectedGeminiFile = null;

/* ==============================
   SEGURIDAD BÁSICA
============================== */

function escapeHTML(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
function cleanGeminiText(text) {
    return String(text || "")
        .replaceAll("**", "")
        .replaceAll("###", "")
        .replaceAll("##", "")
        .replaceAll("#", "")
        .replaceAll("*", "")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
}

function formatGeminiSimpleText(text) {
    const cleanText = cleanGeminiText(text);

    const lines = cleanText
        .split("\n")
        .map(line => line.trim())
        .filter(line => line.length > 0);

    return lines.map(line => {
        const isTitle = /^(1\.|2\.|3\.|4\.|5\.)/.test(line);

        if (isTitle) {
            return `<h4 class="gemini-text-title">${escapeHTML(line)}</h4>`;
        }

        return `<p>${escapeHTML(line)}</p>`;
    }).join("");
}

/* ==============================
   CARRUSEL
============================== */

let currentSlide = 0;
const slides = document.querySelectorAll(".slide");

if (slides.length > 0) {
    setInterval(() => {
        slides[currentSlide].classList.remove("active");
        currentSlide = (currentSlide + 1) % slides.length;
        slides[currentSlide].classList.add("active");
    }, 4500);
}

/* ==============================
   SELECCIÓN DE IMÁGENES MEDPAPAYA
============================== */

imageInput.addEventListener("change", () => {
    const files = Array.from(imageInput.files);

    if (files.length === 0) {
        selectedFiles = [];
        updateFileCounter();
        renderPreviews();
        return;
    }

    if (files.length > MAX_FILES) {
        alert(`Solo puedes subir un máximo de ${MAX_FILES} imágenes.`);
        imageInput.value = "";
        selectedFiles = [];
        updateFileCounter();
        renderPreviews();
        return;
    }

    const invalidFiles = files.filter(file => !file.type.startsWith("image/"));

    if (invalidFiles.length > 0) {
        alert("Todos los archivos deben ser imágenes.");
        imageInput.value = "";
        selectedFiles = [];
        updateFileCounter();
        renderPreviews();
        return;
    }

    selectedFiles = files;
    updateFileCounter();
    renderPreviews();
});

function updateFileCounter() {
    if (selectedFiles.length === 0) {
        fileCounter.textContent = "No hay imágenes seleccionadas.";
    } else {
        fileCounter.textContent = `${selectedFiles.length} imagen(es) seleccionada(s).`;
    }
}

function renderPreviews() {
    previewContainer.innerHTML = "";

    selectedFiles.forEach(file => {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.alt = file.name;
        previewContainer.appendChild(img);
    });
}

/* ==============================
   ANÁLISIS PRINCIPAL MEDPAPAYA
============================== */

analyzeBtn.addEventListener("click", async () => {
    if (selectedFiles.length === 0) {
        alert("Debes seleccionar al menos una imagen.");
        return;
    }

    resultsContainer.innerHTML = "";
    loadingBox.classList.remove("hidden");
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analizando...";

    for (const file of selectedFiles) {
        await analyzeImage(file);
    }

    loadingBox.classList.add("hidden");
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analizar imágenes";

    renderHistory();
});

async function analyzeImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const message = errorData?.detail || "No se pudo analizar la imagen.";
            throw new Error(message);
        }

        const data = await response.json();

        renderResultCard(file, data);
        saveToHistory(data);

    } catch (error) {
        renderErrorCard(file, error.message);
    }
}

/* ==============================
   TARJETAS DE RESULTADO
============================== */

function renderResultCard(file, data) {
    const prediction = data.prediction || {};

    const imageUrl = URL.createObjectURL(file);

    const className = prediction.class_name || "No identificado";
    const label = prediction.label || className;
    const confidence = prediction.confidence ?? "N/D";
    const recommendation =
        prediction.recommendation ||
        "No se recibió una recomendación desde la API.";

    const probabilities = Array.isArray(prediction.probabilities)
        ? prediction.probabilities
        : [];

    const numericConfidence = Number(confidence);

    const isLowConfidence =
        !Number.isNaN(numericConfidence) &&
        numericConfidence < LOW_CONFIDENCE_LIMIT;

    const card = document.createElement("article");
    card.className = "result-card";

    card.innerHTML = `
        <img src="${imageUrl}" alt="${escapeHTML(data.filename || file.name)}">

        <h3>${escapeHTML(label)}</h3>

        <span class="confidence">
            Confianza: ${escapeHTML(String(confidence))}%
        </span>

        <p>
            <strong>Archivo:</strong>
            ${escapeHTML(data.filename || file.name)}
        </p>

        <p>
            <strong>Clase técnica:</strong>
            ${escapeHTML(className)}
        </p>

        <p>
            <strong>Recomendación:</strong>
            ${escapeHTML(recommendation)}
        </p>

        ${
            isLowConfidence
                ? `
                    <div class="gemini-warning">
                        <p>
                            <strong>Resultado inferior al ${LOW_CONFIDENCE_LIMIT}%.</strong>
                            Intenta con Gemini para obtener una segunda opinión orientativa.
                        </p>

                        <button type="button" class="gemini-jump-btn">
                            Usar esta imagen en Gemini
                        </button>
                    </div>
                `
                : ""
        }

        ${
            probabilities.length > 0
                ? `
                    <div class="prob-list">
                        ${probabilities.map(item => `
                            <div class="prob-item">
                                <span>${escapeHTML(item.label || item.class_name || "Clase")}</span>
                                <strong>${escapeHTML(String(item.probability ?? "N/D"))}%</strong>
                            </div>
                        `).join("")}
                    </div>
                `
                : `
                    <div class="prob-list">
                        <div class="prob-item">
                            <span>Probabilidades detalladas</span>
                            <strong>No disponibles</strong>
                        </div>
                    </div>
                `
        }
    `;

    resultsContainer.appendChild(card);

    const geminiJumpButton = card.querySelector(".gemini-jump-btn");

    if (geminiJumpButton) {
        geminiJumpButton.addEventListener("click", () => {
            setGeminiFile(file);

            const geminiSection = document.getElementById("gemini-opinion");

            if (geminiSection) {
                geminiSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    }
}

function renderErrorCard(file, message) {
    const card = document.createElement("article");
    card.className = "result-card error-card";

    card.innerHTML = `
        <h3>Error al analizar</h3>
        <p><strong>Archivo:</strong> ${escapeHTML(file.name)}</p>
        <p>${escapeHTML(message)}</p>
    `;

    resultsContainer.appendChild(card);
}

/* ==============================
   MÓDULO APARTE DE GEMINI
============================== */

if (geminiImageInput) {
    geminiImageInput.addEventListener("change", () => {
        const file = geminiImageInput.files[0];

        if (!file) {
            clearGeminiModule();
            return;
        }

        if (!file.type.startsWith("image/")) {
            alert("El archivo para Gemini debe ser una imagen.");
            geminiImageInput.value = "";
            clearGeminiModule();
            return;
        }

        setGeminiFile(file);
    });
}

if (geminiAnalyzeBtn) {
    geminiAnalyzeBtn.addEventListener("click", async () => {
        if (!selectedGeminiFile) {
            alert("Primero selecciona una imagen para analizar con Gemini.");
            return;
        }

        await analyzeWithGemini(selectedGeminiFile);
    });
}

if (geminiClearBtn) {
    geminiClearBtn.addEventListener("click", () => {
        clearGeminiModule();
    });
}

function setGeminiFile(file) {
    selectedGeminiFile = file;

    if (geminiFileCounter) {
        geminiFileCounter.textContent = `Imagen seleccionada para Gemini: ${file.name}`;
    }

    if (geminiPreviewContainer) {
        geminiPreviewContainer.innerHTML = "";

        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.alt = file.name;

        geminiPreviewContainer.appendChild(img);
    }

    if (geminiStandaloneResult) {
        geminiStandaloneResult.innerHTML = "";
    }
}

function clearGeminiModule() {
    selectedGeminiFile = null;

    if (geminiImageInput) {
        geminiImageInput.value = "";
    }

    if (geminiFileCounter) {
        geminiFileCounter.textContent = "No hay imagen seleccionada para Gemini.";
    }

    if (geminiPreviewContainer) {
        geminiPreviewContainer.innerHTML = "";
    }

    if (geminiStandaloneResult) {
        geminiStandaloneResult.innerHTML = "";
    }

    if (geminiLoadingBox) {
        geminiLoadingBox.classList.add("hidden");
    }

    if (geminiAnalyzeBtn) {
        geminiAnalyzeBtn.disabled = false;
        geminiAnalyzeBtn.textContent = "Analizar con Gemini";
    }
}

async function analyzeWithGemini(file) {
    const formData = new FormData();
    formData.append("file", file);

    if (geminiLoadingBox) {
        geminiLoadingBox.classList.remove("hidden");
    }

    if (geminiStandaloneResult) {
        geminiStandaloneResult.innerHTML = "";
    }

    if (geminiAnalyzeBtn) {
        geminiAnalyzeBtn.disabled = true;
        geminiAnalyzeBtn.textContent = "Analizando con Gemini...";
    }

    try {
        const response = await fetch(GEMINI_API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const message =
                errorData?.detail ||
                "No se pudo obtener la segunda opinión con Gemini.";

            throw new Error(message);
        }

        const data = await response.json();

        const formattedAnalysis = formatGeminiSimpleText(
    data.analysis || "Gemini no devolvió análisis."
);

if (geminiStandaloneResult) {
    geminiStandaloneResult.innerHTML = `
        <div class="gemini-result-header">
            <span class="gemini-chip">Segunda opinión</span>
            <h3>Resultado de Gemini</h3>
        </div>

        <div class="gemini-result-meta">
            <p><strong>Archivo:</strong> ${escapeHTML(data.filename || file.name)}</p>
            <p><strong>Modelo usado:</strong> ${escapeHTML(data.model || "Gemini")}</p>
        </div>

        <div class="gemini-analysis-text">
            ${formattedAnalysis}
        </div>
    `;
}

    } catch (error) {
        console.error("Error con Gemini:", error);

        if (geminiStandaloneResult) {
            geminiStandaloneResult.innerHTML = `
                <h3>Error al consultar Gemini</h3>
                <p>${escapeHTML(error.message)}</p>
            `;
        }

    } finally {
        if (geminiLoadingBox) {
            geminiLoadingBox.classList.add("hidden");
        }

        if (geminiAnalyzeBtn) {
            geminiAnalyzeBtn.disabled = false;
            geminiAnalyzeBtn.textContent = "Analizar con Gemini";
        }
    }
}

/* ==============================
   HISTORIAL
============================== */

function saveToHistory(data) {
    const prediction = data.prediction || {};

    const item = {
        filename: data.filename,
        label: prediction.label || prediction.class_name || "No identificado",
        confidence: prediction.confidence ?? "N/D",
        className: prediction.class_name || "No identificado",
        date: new Date().toLocaleString()
    };

    const history = JSON.parse(localStorage.getItem("medpapaya_history")) || [];

    history.unshift(item);

    const limitedHistory = history.slice(0, 10);

    localStorage.setItem("medpapaya_history", JSON.stringify(limitedHistory));
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem("medpapaya_history")) || [];

    historyContainer.innerHTML = "";

    if (history.length === 0) {
        historyContainer.innerHTML = `
            <div class="history-item">
                <span>No hay análisis registrados todavía.</span>
            </div>
        `;
        return;
    }

    history.forEach(item => {
        const row = document.createElement("div");
        row.className = "history-item";

        row.innerHTML = `
            <div>
                <strong>${escapeHTML(item.label)}</strong>
                <p>${escapeHTML(item.filename)}</p>
            </div>

            <div>
                <strong>${escapeHTML(String(item.confidence))}%</strong>
                <p>${escapeHTML(item.date)}</p>
            </div>
        `;

        historyContainer.appendChild(row);
    });
}

/* ==============================
   LIMPIAR
============================== */

clearBtn.addEventListener("click", () => {
    selectedFiles = [];
    imageInput.value = "";
    previewContainer.innerHTML = "";
    resultsContainer.innerHTML = "";
    updateFileCounter();
});

renderHistory();