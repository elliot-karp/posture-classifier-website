
import { setModelPath } from "./tensor.js";
let useAngles = true;
export { useAngles };
document.addEventListener("DOMContentLoaded", () => {
  const openSettingsButton = document.getElementById("open-settings");
  const closeSettingsButton = document.getElementById("close-settings");
  const settingsPopup = document.getElementById("settings-popup");
  const audioQueueToggle = document.getElementById("audio-queue-toggle");
  const modelDropdown = document.getElementById("model-dropdown");

  const savedAudioQueueSetting = localStorage.getItem("audioQueueEnabled") === "true";
  const savedModelType = localStorage.getItem("selectedModelType") || "angles_model";

  audioQueueToggle.checked = savedAudioQueueSetting;
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
  });

  modelDropdown.addEventListener("change", (event) => {
    const selectedModelType = event.target.value;
    localStorage.setItem("selectedModelType", selectedModelType);

    const baseModelPath = "./assets/models/";
    const newModelPath = `${baseModelPath}${selectedModelType}`;
    console.log(`Switching model to: ${newModelPath}`);

    setModelPath(newModelPath);
  });
});