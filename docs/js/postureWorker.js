let isAudioEnabled = true;
let BAD_POSTURE_THRESHOLD = 3000;
let badPostureStart = null;
console.log("loaded worker")
self.onmessage = (event) => {


  if (event.data.type === "updateSettings") {
    isAudioEnabled = event.data.data.isAudioEnabled;
    BAD_POSTURE_THRESHOLD = event.data.data.BAD_POSTURE_THRESHOLD;
    console.log("Worker settings updated:", { isAudioEnabled, BAD_POSTURE_THRESHOLD });
  }

  if (event.data.type === "postureData") {
    const { prediction } = event.data.data;

    if (prediction === 0) {
      if (!badPostureStart) badPostureStart = Date.now();
      const elapsedTime = Date.now() - badPostureStart;
      if (isAudioEnabled && elapsedTime >= BAD_POSTURE_THRESHOLD) {
        badPostureStart = null;
        postMessage({ type: "playAlert" }); 
    
      }
    } else if (prediction === 1) {
      badPostureStart = null;
    }
  }
};


