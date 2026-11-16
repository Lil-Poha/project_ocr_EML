// Логика страницы: выбор фото, отправка в gateway, показ результата.
// Без фреймворков — обычный JavaScript, который понимает любой браузер.
"use strict";

// Должно совпадать с настройкой MAX_UPLOAD_MB в gateway
const MAX_UPLOAD_MB = 10;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"];
// Адрес относительный: nginx перенаправляет /api/ в gateway
const RECOGNIZE_URL = "/api/recognize";

const $ = (id) => document.getElementById(id);

const ui = {
  upload: $("upload"),
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  uploadError: $("upload-error"),
  workspace: $("workspace"),
  fileMeta: $("file-meta"),
  previewImage: $("preview-image"),
  previewFallback: $("preview-fallback"),
  resultBox: $("result-box"),
  placeholder: $("result-placeholder"),
  loading: $("result-loading"),
  resultText: $("result-text"),
  demoBadge: $("demo-badge"),
  demoNote: $("demo-note"),
  error: $("error"),
  recognizeBtn: $("recognize-btn"),
  copyBtn: $("copy-btn"),
  resetBtn: $("reset-btn"),
  modelName: $("model-name"),
};

let currentFile = null;
let previewUrl = null;
let activeRequest = null;

function cancelRecognition() {
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  ui.loading.hidden = true;
  ui.resultBox.setAttribute("aria-busy", "false");
  ui.recognizeBtn.disabled = false;
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.hidden = true;
  ui.error.textContent = "";
}

/** Проверка на стороне браузера, чтобы не отправлять заведомо неподходящий файл. */
function validate(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return "Неподдерживаемый формат. Подойдут JPEG, PNG, TIFF, BMP или WebP.";
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `Файл слишком большой (${formatSize(file.size)}). Максимум ${MAX_UPLOAD_MB} МБ.`;
  }
  return null;
}

function resetResult() {
  ui.placeholder.hidden = false;
  ui.loading.hidden = true;
  ui.resultText.hidden = true;
  ui.resultText.textContent = "";
  ui.demoBadge.hidden = true;
  ui.demoNote.hidden = true;
  ui.copyBtn.hidden = true;
  ui.copyBtn.textContent = "Копировать";
  ui.modelName.textContent = "—";
  clearError();
}

function selectFile(file) {
  const problem = validate(file);
  if (problem) {
    // Ошибку показываем там, где пользователь сейчас находится
    if (ui.workspace.hidden) {
      ui.uploadError.textContent = problem;
      ui.uploadError.hidden = false;
    } else {
      showError(problem);
    }
    return;
  }
  ui.uploadError.hidden = true;

  cancelRecognition();
  currentFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);

  ui.previewImage.hidden = false;
  ui.previewFallback.hidden = true;
  ui.previewImage.src = previewUrl;
  ui.fileMeta.textContent = `${file.name} · ${formatSize(file.size)}`;

  resetResult();
  ui.upload.hidden = true;
  ui.workspace.hidden = false;
  ui.recognizeBtn.focus();
}

async function recognize() {
  if (!currentFile) return;

  cancelRecognition();
  const controller = new AbortController();
  activeRequest = controller;
  resetResult();
  ui.placeholder.hidden = true;
  ui.loading.hidden = false;
  ui.resultBox.setAttribute("aria-busy", "true");
  ui.recognizeBtn.disabled = true;

  const form = new FormData();
  form.append("file", currentFile);

  try {
    const response = await fetch(RECOGNIZE_URL, {
      method: "POST", body: form, signal: controller.signal,
    });
    // Сервер может ответить не JSON (например, nginx при сбое) — учитываем это
    const data = await response.json().catch(() => ({}));
    if (activeRequest !== controller) return;

    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : null;
      throw new Error(detail || `Ошибка сервера (${response.status}). Попробуйте ещё раз.`);
    }

    ui.resultText.textContent = data.text || "Текст не найден.";
    ui.resultText.hidden = false;
    ui.copyBtn.hidden = !data.text;
    ui.demoBadge.hidden = !data.stub;
    ui.demoNote.hidden = !data.stub;
    ui.modelName.textContent = data.model || "—";
  } catch (err) {
    if (activeRequest !== controller || err.name === "AbortError") return;
    ui.placeholder.hidden = false;
    // TypeError от fetch означает, что сервер вообще недоступен
    const message = err instanceof TypeError
      ? "Не удалось связаться с сервером. Проверьте, что сервис запущен."
      : err.message;
    showError(message);
  } finally {
    if (activeRequest === controller) {
      activeRequest = null;
      ui.loading.hidden = true;
      ui.resultBox.setAttribute("aria-busy", "false");
      ui.recognizeBtn.disabled = false;
    }
  }
}

async function copyResult() {
  try {
    await navigator.clipboard.writeText(ui.resultText.textContent);
    ui.copyBtn.textContent = "Скопировано";
    setTimeout(() => { ui.copyBtn.textContent = "Копировать"; }, 1600);
  } catch {
    showError("Не удалось скопировать. Выделите текст и скопируйте вручную.");
  }
}

function reset() {
  cancelRecognition();
  currentFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  ui.previewImage.removeAttribute("src");
  ui.fileInput.value = "";
  resetResult();
  ui.workspace.hidden = true;
  ui.upload.hidden = false;
}

// ---------- События ----------

ui.fileInput.addEventListener("change", () => {
  const file = ui.fileInput.files[0];
  if (file) selectFile(file);
});

// Перетаскивание файла в зону загрузки
["dragenter", "dragover"].forEach((type) =>
  ui.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    ui.dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((type) =>
  ui.dropzone.addEventListener(type, () => ui.dropzone.classList.remove("is-dragover"))
);
ui.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) selectFile(file);
});

// Если файл бросили мимо зоны, браузер не должен открывать его вместо страницы
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

// TIFF большинство браузеров показать не может — выводим пояснение вместо картинки
ui.previewImage.addEventListener("error", () => {
  if (!currentFile) return;
  ui.previewImage.hidden = true;
  ui.previewFallback.hidden = false;
});

ui.recognizeBtn.addEventListener("click", recognize);
ui.copyBtn.addEventListener("click", copyResult);
ui.resetBtn.addEventListener("click", reset);
