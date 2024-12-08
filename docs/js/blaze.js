import { queryModel } from "./tensor.js";

let scalingParams = null;

// Load scaling parameters
fetch("scaling_params.json")
  .then((response) => response.json())
  .then((data) => {
    scalingParams = data;
    //console.log("Scaling parameters loaded:", scalingParams);
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
    //match canvas size to video feed
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    // Draw overlays for nose and shoulders
    const landmarks = results.poseLandmarks;
    const nose = landmarks[0];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];

    drawLandmark(nose, "red", canvasCtx);
    drawLandmark(leftShoulder, "blue", canvasCtx);
    drawLandmark(rightShoulder, "blue", canvasCtx);

    // Calculate angles
    const shoulderTilt = calculateAngle(leftShoulder, rightShoulder);
    const forwardSlouchAngle = calculateForwardSlouchAngle(nose, leftShoulder, rightShoulder);
    const neckAngle = calculateThreePointAngle(leftShoulder, nose, rightShoulder);

    const rawInput = [shoulderTilt, forwardSlouchAngle, neckAngle];
    console.log("Raw Input:", rawInput);

    if (!scalingParams) {
      console.error("Scaling parameters not loaded yet. Skipping prediction.");
      return;
    }

    // Scale the input using the loaded parameters
    const scaledInput = rawInput.map((value, index) => {
      const mean = scalingParams.means[index];
      const stdDev = scalingParams.std_devs[index];
      return (value - mean) / stdDev;
    });

    //console.log("Scaled Input:", scaledInput);

    // Pass the scaled input to the TensorFlow.js model
    queryModel(scaledInput).then((prediction) => {
      // Change the canvas border color based on prediction
      if (prediction === 0) {
        canvasElement.style.border = "5px solid red";
      } else if (prediction === 1) {
        canvasElement.style.border = "5px solid green";
      }
      else {
        console.error("Unexpected prediction value:", prediction); 

      }



    });
  });

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await pose.send({ image: videoElement });
    },
    width: 640, // Match the canvas width
    height: 360, // Match the canvas height
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
      noseToMidpointVector.z * verticalVector.z + noseToMidpointVector.y * verticalVector.y;
    const magnitudeNose = Math.sqrt(
      noseToMidpointVector.z ** 2 + noseToMidpointVector.y ** 2
    );
    const magnitudeVertical = Math.sqrt(verticalVector.z ** 2 + verticalVector.y ** 2);

    if (magnitudeNose === 0 || magnitudeVertical === 0) return 0;

    const angleRadians = Math.acos(dotProduct / (magnitudeNose * magnitudeVertical));
    return (angleRadians * 180) / Math.PI;
  }
});