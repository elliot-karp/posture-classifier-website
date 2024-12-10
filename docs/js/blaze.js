import { queryModel, selectedModelPath } from "./tensor.js";
import { useAngles } from "./settings.js";


export const postureWorker = new Worker("./js/postureWorker.js");


let scalingParams = null;
let badPostureStart = null;
let isAudioEnabled = localStorage.getItem("audioQueueEnabled") === "true";
let BAD_POSTURE_THRESHOLD = (localStorage.getItem("alertTime") || 3) * 1000;


postureWorker.postMessage({
  type: "updateSettings",
  data: { isAudioEnabled, BAD_POSTURE_THRESHOLD },
});


document.addEventListener("settingsUpdated", () => {
  isAudioEnabled = localStorage.getItem("audioQueueEnabled") === "true";
  BAD_POSTURE_THRESHOLD = (localStorage.getItem("alertTime") || 3) * 1000;
  postureWorker.postMessage({
    type: "updateSettings",
    data: { isAudioEnabled, BAD_POSTURE_THRESHOLD },
  });
  console.log("Settings updated:", { isAudioEnabled, BAD_POSTURE_THRESHOLD });
});


postureWorker.onmessage = (event) => {
  if (event.data.type === "playAlert") {

    if (isAudioEnabled) {
      const audio = new Audio("./assets/fix_posture.mp3");
      audio.play().catch((error) => console.error("Audio playback error:", error));
    }

  }
};


fetch(`${selectedModelPath}/scaling_params.json`)
  .then((response) => response.json())
  .then((data) => (scalingParams = data))
  .catch((error) => console.error("Error loading scaling parameters:", error));


  document.addEventListener("DOMContentLoaded", () => {
    const videoElement = document.getElementById("video");
    const canvasElement = document.getElementById("output");
    const canvasCtx = canvasElement.getContext("2d");
  
    // Use the existing hidden video element from the HTML
    const hiddenVideoElement = document.getElementById("pip-video");
  
    // Set up the canvas stream for PiP
    const canvasStream = canvasElement.captureStream();
    hiddenVideoElement.srcObject = canvasStream;
  
    // Use the existing button from HTML
    const pipButton = document.getElementById("pip-button");
  
    // Add event listener for PiP button
    pipButton.addEventListener("click", async () => {
      try {
        if (document.pictureInPictureElement) {
          await document.exitPictureInPicture();
        } else {
          await hiddenVideoElement.play();
          await hiddenVideoElement.requestPictureInPicture();
        }
      } catch (error) {
        console.error("Error enabling PiP:", error);
      }
    });
  
    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
  
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  
    pose.onResults((results) => {
      processPoseResults(results, canvasElement, canvasCtx);
    });
  
    const camera = new Camera(videoElement, {
      onFrame: async () => await pose.send({ image: videoElement }),
      width: 640,
      height: 360,
    });
  
    camera.start();
  });


let lastPrediction = 0; 

function processPoseResults(results, canvasElement, canvasCtx) {
  if (!results.poseLandmarks) return;

  const videoElement = document.getElementById("video");


  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  const landmarks = results.poseLandmarks;

 
  drawLandmark(landmarks[0], "red", canvasCtx);    
  drawLandmark(landmarks[11], "blue", canvasCtx); 
  drawLandmark(landmarks[12], "blue", canvasCtx);  

 
  drawBorder(lastPrediction, canvasElement, canvasCtx);

 
  if (!scalingParams) {
    console.error("Scaling parameters not loaded yet. Skipping prediction.");
    return;
  }

  const rawInput = useAngles ? calculateAngles(landmarks) : calculatePoints(landmarks);
  const scaledInput = rawInput.map((value, index) => {
    const mean = scalingParams.means[index];
    const stdDev = scalingParams.std_devs[index];
    return (value - mean) / stdDev;
  });

  
  queryModel(scaledInput).then((prediction) => {
    lastPrediction = prediction;
    postureWorker.postMessage({ type: "postureData", data: { prediction } });
  });
}

function drawBorder(prediction, canvasElement, canvasCtx) {
  
  const borderWidth = 30;
  let borderColor = prediction ? "green" : "red";

  canvasCtx.save();
  canvasCtx.lineWidth = borderWidth;
  canvasCtx.strokeStyle = borderColor;
  canvasCtx.strokeRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.restore();

  canvasElement.style.transition = "border-color 0.01s";
  canvasElement.style.border = `6px solid ${borderColor}`;
}


function drawLandmark(landmark, color, ctx) {
  if (landmark.visibility > 0.5) {
    ctx.beginPath();
    ctx.arc(landmark.x * ctx.canvas.width, landmark.y * ctx.canvas.height, 5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }
}



function calculateAngles(landmarks) {
  const shoulderTilt = calculateAngle(landmarks[11], landmarks[12]);
  const forwardSlouchAngle = calculateForwardSlouchAngle(
    landmarks[0],
    landmarks[11],
    landmarks[12]
  );
  const neckAngle = calculateThreePointAngle(landmarks[11], landmarks[0], landmarks[12]);
  return [shoulderTilt, forwardSlouchAngle, neckAngle];
}

function calculatePoints(landmarks) {
  return [
    landmarks[0].x, landmarks[0].y, landmarks[0].z, 
    landmarks[11].x, landmarks[11].y, landmarks[11].z, 
    landmarks[12].x, landmarks[12].y, landmarks[12].z, 
  ];
}

function calculateAngle(p1, p2) {
  const deltaX = p2.x - p1.x;
  const deltaY = p2.y - p1.y;
  return Math.atan2(deltaY, deltaX) * (180 / Math.PI);
}

function calculateThreePointAngle(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const dotProduct = ba.x * bc.x + ba.y * bc.y;
  const magnitudeBA = Math.sqrt(ba.x ** 2 + ba.y ** 2);
  const magnitudeBC = Math.sqrt(bc.x ** 2 + bc.y ** 2);
  return Math.acos(dotProduct / (magnitudeBA * magnitudeBC)) * (180 / Math.PI);
}

function calculateForwardSlouchAngle(nose, leftShoulder, rightShoulder) {
  const midpoint = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
    z: (leftShoulder.z + rightShoulder.z) / 2,
  };
  const vector = { y: nose.y - midpoint.y, z: nose.z - midpoint.z };
  const vertical = { y: 1, z: 0 };
  const dotProduct = vector.y * vertical.y + vector.z * vertical.z;
  const magnitudeVector = Math.sqrt(vector.y ** 2 + vector.z ** 2);
  const magnitudeVertical = Math.sqrt(vertical.y ** 2 + vertical.z ** 2);
  return Math.acos(dotProduct / (magnitudeVector * magnitudeVertical)) * (180 / Math.PI);
}