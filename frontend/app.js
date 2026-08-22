const fileInput = document.getElementById("file-input");
const chooseButton = document.getElementById("choose-button");
const dropZone = document.getElementById("drop-zone");

const previewSection =
  document.getElementById("preview-section");

const imagePreview =
  document.getElementById("image-preview");

const analyzeButton =
  document.getElementById("analyze-button");

const result =
  document.getElementById("result");

const fileName =
  document.getElementById("file-name");

let selectedFile = null;


/* OPEN FILE PICKER */

chooseButton.addEventListener("click", (event) => {
  event.stopPropagation();

  fileInput.click();
});


dropZone.addEventListener("click", () => {
  fileInput.click();
});


/* NORMAL FILE SELECTION */

fileInput.addEventListener("change", () => {

  if (!fileInput.files.length) {
    return;
  }

  handleFile(fileInput.files[0]);

});


/* DRAG AND DROP */

dropZone.addEventListener("dragover", (event) => {

  event.preventDefault();

  dropZone.classList.add("dragging");

});


dropZone.addEventListener("dragleave", () => {

  dropZone.classList.remove("dragging");

});


dropZone.addEventListener("drop", (event) => {

  event.preventDefault();

  dropZone.classList.remove("dragging");

  const file = event.dataTransfer.files[0];

  if (file) {
    handleFile(file);
  }

});


/* HANDLE SELECTED IMAGE */

function handleFile(file) {

  const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/webp"
  ];

  if (!allowedTypes.includes(file.type)) {

    alert(
      "Please upload a JPG, PNG, or WebP image."
    );

    return;
  }


  const maxSize =
    10 * 1024 * 1024;


  if (file.size > maxSize) {

    alert(
      "Image must be smaller than 10MB."
    );

    return;
  }


  selectedFile = file;

  fileName.textContent =
    file.name;


  const imageURL =
    URL.createObjectURL(file);


  imagePreview.src =
    imageURL;


  previewSection.classList.remove(
    "hidden"
  );


  result.textContent = "";

}


/* CALL FASTAPI */

analyzeButton.addEventListener(
  "click",
  async () => {

    if (!selectedFile) {
      return;
    }


    analyzeButton.disabled = true;

    analyzeButton.textContent =
      "Analyzing...";

    result.textContent = "";


    const formData =
      new FormData();


    formData.append(
      "file",
      selectedFile
    );


    try {

      const response = await fetch(
        "https://am-i-a-chicken.onrender.com/api/predict",
        {
          method: "POST",
          body: formData
        }
      );


      if (!response.ok) {
        throw new Error(
          "Prediction failed"
        );
      }


      const data =
        await response.json();


      console.log(data);


      /*
       Change these fields if your
       FastAPI response uses
       different names.
      */

      result.textContent =
        `${data.prediction} — ${Math.round(
          data.confidence * 100
        )}% confidence`;


    } catch (error) {

      console.error(error);

      result.textContent =
        "Unable to reach the prediction server.";

    } finally {

      analyzeButton.disabled = false;

      analyzeButton.textContent =
        "Analyze Image";

    }

  }
);