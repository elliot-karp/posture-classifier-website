import { queryModel, selectedModelPath } from "./tensor.js";
import { useAngles } from "./settings.js"

let scalingParams = null;
const audio = new Audio("./assets/fix_posture.mp3");
const BAD_POSTURE_THRESHOLD = 10000;
let badPostureStart = null;  


 
fetch(`${selectedModelPath}/scaling_params.json`)
  .then((response) => response.json())
  .then((data) => {
    scalingParams = data;
  })
  .catch((error) => {
    console.error("Error loading scaling parameters:", error);
  });

document.addEventListener("DOMContentLoaded", () => {
  const videoElement = document.getElementById("video");
  const canvasElement = document.getElementById("output");
  const canvasCtx = canvasElement.getContext("2d");

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
    if (!results.poseLandmarks) return;

     canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    const landmarks = results.poseLandmarks;

     drawLandmark(landmarks[0], "red", canvasCtx); // Nose
    drawLandmark(landmarks[11], "blue", canvasCtx); // Left Shoulder
    drawLandmark(landmarks[12], "blue", canvasCtx); // Right Shoulder

    let rawInput;
    if (useAngles) {
    
      const shoulderTilt = calculateAngle(landmarks[11], landmarks[12]);
      const forwardSlouchAngle = calculateForwardSlouchAngle(
        landmarks[0],
        landmarks[11],
        landmarks[12]
      );
      const neckAngle = calculateThreePointAngle(
        landmarks[11],
        landmarks[0],
        landmarks[12]
      );
      rawInput = [shoulderTilt, forwardSlouchAngle, neckAngle];
    } else {
       rawInput = [
        landmarks[0].x, landmarks[0].y, landmarks[0].z, // Nose
        landmarks[1].x, landmarks[1].y, landmarks[1].z, // Left Eye
        landmarks[2].x, landmarks[2].y, landmarks[2].z, // Right Eye
        landmarks[11].x, landmarks[11].y, landmarks[11].z, // Left Shoulder
        landmarks[12].x, landmarks[12].y, landmarks[12].z // Right Shoulder
      ];
    }

    if (!scalingParams) {
      console.error("Scaling parameters not loaded yet. Skipping prediction.");
      return;
    }

     const scaledInput = rawInput.map((value, index) => {
      const mean = scalingParams.means[index];
      const stdDev = scalingParams.std_devs[index];
      return (value - mean) / stdDev;
    });

    queryModel(scaledInput).then((prediction) => {
      if (prediction === 0) {
        canvasElement.style.border = "6px solid red";

        const isAudioEnabled =
          localStorage.getItem("audioQueueEnabled") === "true";

         if (!badPostureStart) {
          badPostureStart = Date.now();
        }

         const elapsedTime = Date.now() - badPostureStart;
        if (isAudioEnabled && elapsedTime >= BAD_POSTURE_THRESHOLD) {
          audio.play();
          console.log("Bad posture for x seconds");
          badPostureStart = null;  
        }
      } else if (prediction === 1) {
        canvasElement.style.border = "6px solid green";

         badPostureStart = null;
      } else {
        console.error("Unexpected prediction value:", prediction);
      }
    });
  });

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await pose.send({ image: videoElement });
    },
    width: 640,  
    height: 360,  
  });

  camera.start();

  function drawLandmark(landmark, color, ctx) {
    if (landmark.visibility > 0.5) {
      ctx.beginPath();
      ctx.arc(
        landmark.x * canvasElement.width,
        landmark.y * canvasElement.height,
        5,
        0,
        2 * Math.PI
      );
      ctx.fillStyle = color;
      ctx.fill();
    }
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

    if (magnitudeBA === 0 || magnitudeBC === 0) return 0;

    const angle = Math.acos(dotProduct / (magnitudeBA * magnitudeBC));
    return (angle * 180) / Math.PI;
  }

  function calculateForwardSlouchAngle(nose, leftShoulder, rightShoulder) {
    const shoulderMidpoint = {
      x: (leftShoulder.x + rightShoulder.x) / 2.0,
      y: (leftShoulder.y + rightShoulder.y) / 2.0,
      z: (leftShoulder.z + rightShoulder.z) / 2.0,
    };

    const noseToMidpointVector = {
      z: nose.z - shoulderMidpoint.z,
      y: nose.y - shoulderMidpoint.y,
    };

    const verticalVector = { z: 0, y: 1 };

    const dotProduct =
      noseToMidpointVector.z * verticalVector.z +
      noseToMidpointVector.y * verticalVector.y;
    const magnitudeNose = Math.sqrt(
      noseToMidpointVector.z ** 2 + noseToMidpointVector.y ** 2
    );
    const magnitudeVertical = Math.sqrt(
      verticalVector.z ** 2 + verticalVector.y ** 2
    );

    if (magnitudeNose === 0 || magnitudeVertical === 0) return 0;

    const angleRadians = Math.acos(
      dotProduct / (magnitudeNose * magnitudeVertical)
    );
    return (angleRadians * 180) / Math.PI;
  }
});