const uploadInput = document.getElementById("upload-input");
const uploadButton = document.getElementById("upload-button");
const analyzeButton = document.getElementById("analyze-button");

const preview = document.getElementById("preview");
const photoStatus = document.getElementById("photo-status");
const result = document.getElementById("result");

let selectedFile = null;
let previewUrl = null;

uploadButton.addEventListener("click", () => {
    uploadInput.click();
});

uploadInput.addEventListener("change", () => {
    const file = uploadInput.files[0];

    if (!file || !file.type.startsWith("image/")) {
        selectedFile = null;
        analyzeButton.disabled = true;

        preview.style.display = "none";
        photoStatus.style.display = "block";
        photoStatus.textContent = "Please select a valid image.";

        return;
    }

    selectedFile = file;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    preview.style.display = "block";

    photoStatus.style.display = "none";

    analyzeButton.disabled = false;
    result.textContent = "Photo ready.";
});

analyzeButton.addEventListener("click", async () => {
    if (!selectedFile) {
        return;
    }

    analyzeButton.disabled = true;
    result.textContent = "ANALYZING...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);

            throw new Error(
                errorData?.detail || `API error: ${response.status}`
            );
        }

        const data = await response.json();

        const prediction = data.prediction.toUpperCase();
        const confidence = Math.round(data.confidence * 100);

        result.textContent = `${confidence}% ${prediction}`;
    } catch (error) {
        console.error(error);
        result.textContent = `Prediction failed: ${error.message}`;
    } finally {
        analyzeButton.disabled = false;
    }
});