
let useAngles = true;
export { useAngles };
import { loadModel } from "./tensor.js";
import { postureWorker} from "./blaze.js"

document.addEventListener("DOMContentLoaded", () => {
  const openSettingsButton = document.getElementById("open-settings");
  const closeSettingsButton = document.getElementById("close-settings");
  const settingsPopup = document.getElementById("settings-popup");
  const audioQueueToggle = document.getElementById("audio-queue-toggle");
  const alertTimeInput = document.getElementById("alert-time");
  const modelDropdown = document.getElementById("model-dropdown");

  const savedAudioQueueSetting = localStorage.getItem("audioQueueEnabled") === "true";
  const savedAlertTime = parseInt(localStorage.getItem("alertTime")) || 8;
  const savedModelType = localStorage.getItem("selectedModelType") || "angles_model";

  audioQueueToggle.checked = savedAudioQueueSetting;
  alertTimeInput.value = savedAlertTime;
  modelDropdown.value = savedModelType;


  openSettingsButton.addEventListener("click", () => {
    settingsPopup.classList.remove("hidden"); 
  });


  closeSettingsButton.addEventListener("click", () => {
    settingsPopup.classList.add("hidden"); 
  });


audioQueueToggle.addEventListener("change", (event) => {
  const isAudioEnabled = event.target.checked;
  localStorage.setItem("audioQueueEnabled", isAudioEnabled);
  console.log("Audio Queue Enabled:", isAudioEnabled);

  document.dispatchEvent(new Event("settingsUpdated"));
});

alertTimeInput.addEventListener("change", (event) => {
  const alertTime = parseInt(event.target.value);
  if (alertTime >= 1) {
    localStorage.setItem("alertTime", alertTime);
    console.log("Alert Time Interval (seconds):", alertTime);

    document.dispatchEvent(new Event("settingsUpdated"));
  } else {
    console.error("Invalid alert time specified.");
  }
});

modelDropdown.addEventListener("change", (event) => {
  const selectedModelType = event.target.value;
  localStorage.setItem("selectedModelType", selectedModelType);
  console.log("Selected Model Type:", selectedModelType);

  document.dispatchEvent(new Event("settingsUpdated"));

  switchModel(selectedModelType);
});


  function switchModel(modelType) {
    const baseModelPath = "./assets/models/";
    const selectedModelPath = `${baseModelPath}${modelType}`;
    console.log(`Switching model to: ${selectedModelPath}`);
    window.selectedModelPath = selectedModelPath;

    window.modelPromise = loadModel();
  }
});

document.addEventListener("settingsUpdated", () => {
  const isAudioEnabled = localStorage.getItem("audioQueueEnabled") === "true";
  const BAD_POSTURE_THRESHOLD = (localStorage.getItem("alertTime") || 5) * 1000;

  postureWorker.postMessage({
    type: "updateSettings",
    data: {
      isAudioEnabled,
      BAD_POSTURE_THRESHOLD,
    },
  });

  console.log("Settings updated in the worker");
});

const canvasElement = document.getElementById("output");
const pipVideoElement = document.getElementById("pip-video");
const pipButton = document.getElementById("pip-button"); 


const canvasStream = canvasElement.captureStream();


pipVideoElement.srcObject = canvasStream;

pipButton.addEventListener("click", async () => {
  try {
    if (document.pictureInPictureElement) {
      await document.exitPictureInPicture(); 
    } else {
      await pipVideoElement.requestPictureInPicture();
    }
  } catch (error) {
    console.error("Error enabling PiP:", error);
  }
});