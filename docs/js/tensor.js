 export let selectedModelPath = "./assets/models/angles_model";

 export async function loadModel() {
  try {
    console.log(`Loading TensorFlow.js model from ${selectedModelPath}/model.json...`);
    const model = await tf.loadLayersModel(`${selectedModelPath}/model.json`);
    console.log("Model loaded successfully :)", model);
    return model;
  } catch (error) {
    console.error("Error loading the model:", error);
    return null;
  }
}

 export function setModelPath(newPath) {
  selectedModelPath = newPath;
  console.log(`Model path set to: ${selectedModelPath}`);
  return loadModel();
}

 export let modelPromise = loadModel();

 export async function queryModel(blazeInput) {
  const model = await modelPromise;  
  if (!model) {
    console.error("Model not loaded, skipping prediction.");
    return null;
  }

  try {
    const inputTensor = tf.tensor2d([blazeInput]);
    const output = model.predict(inputTensor);
    const probabilities = tf.softmax(output);
    const predictedClassTensor = probabilities.argMax(1);
    const predictedClass = (await predictedClassTensor.data())[0];

    inputTensor.dispose();
    output.dispose();
    probabilities.dispose();
    predictedClassTensor.dispose();

    return predictedClass;
  } catch (error) {
    console.error("Error during model prediction:", error);
    return null;
  }
}