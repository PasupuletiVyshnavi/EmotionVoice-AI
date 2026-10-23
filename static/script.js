
const fileInput = document.getElementById("audioFile");
const dropzone = document.getElementById("dropzone");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const audioPreview = document.getElementById("audioPreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const removeFile = document.getElementById("removeFile");
const message = document.getElementById("message");

let selectedFile = null;
let previewUrl = null;

function showMessage(text) {
    message.textContent = text;
}

function chooseFile(file) {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".wav")) {
        showMessage("Please choose a WAV audio file.");
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        showMessage("File is too large. Maximum size is 20 MB.");
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
    fileInfo.classList.remove("hidden");
    audioPreview.classList.remove("hidden");

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    audioPreview.src = previewUrl;

    analyzeBtn.disabled = false;
    showMessage("");

    document.getElementById("result").classList.add("hidden");
    document.getElementById("emptyState").classList.remove("hidden");
}

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));

["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", event => {
    const file = event.dataTransfer.files[0];
    chooseFile(file);
});

removeFile.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    audioPreview.classList.add("hidden");
    audioPreview.removeAttribute("src");
    analyzeBtn.disabled = true;
    showMessage("");

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
});

const emotionIcons = {
    neutral: "◉",
    calm: "☁",
    happy: "☺",
    sad: "☂",
    angry: "⚡",
    fearful: "!",
    disgust: "◌",
    surprised: "✦"
};

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.querySelector("span").textContent = "…";
    showMessage("Analyzing audio. Please wait.");

    const formData = new FormData();
    formData.append("audio", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        document.getElementById("emptyState").classList.add("hidden");
        document.getElementById("result").classList.remove("hidden");

        document.getElementById("emotionName").textContent = data.emotion;
        document.getElementById("emotionIcon").textContent =
            emotionIcons[data.emotion] || "♫";

        document.getElementById("confidence").textContent =
            data.confidence + "%";

        document.getElementById("confidenceBar").style.width =
            Math.max(0, Math.min(100, data.confidence)) + "%";

        const scoreList = document.getElementById("scoreList");
        scoreList.replaceChildren();

        Object.entries(data.scores)
            .sort((a, b) => b[1] - a[1])
            .forEach(([emotion, score]) => {
                const row = document.createElement("div");
                row.className = "score-row";

                const name = document.createElement("span");
                name.className = "score-name";
                name.textContent = emotion;

                const track = document.createElement("div");
                track.className = "score-track";

                const bar = document.createElement("div");
                bar.className = "score-bar";
                bar.style.width = Math.max(0, Math.min(100, score)) + "%";
                track.appendChild(bar);

                const value = document.createElement("span");
                value.className = "score-value";
                value.textContent = score + "%";

                row.append(name, track, value);
                scoreList.appendChild(row);
            });

        showMessage("Analysis complete.");
    } catch (error) {
        showMessage(error.message);
    } finally {
        analyzeBtn.disabled = !selectedFile;
        analyzeBtn.querySelector("span").textContent = "→";
    }
});

