// Default model path
export let selectedModelPath = "./assets/models/angles_model";

// Function to load the model dynamically based on selectedModelPath
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

// Function to update the selected model path and reload the model
export function setModelPath(newPath) {
  selectedModelPath = newPath;
  console.log(`Model path set to: ${selectedModelPath}`);
  return loadModel();
}

// Load the initial model and make the promise accessible globally
export let modelPromise = loadModel();

// Function to dynamically query the model
export async function queryModel(blazeInput) {
  const model = await modelPromise; // Ensure the latest model is used
  if (!model) {
    console.error("Model not loaded, skipping prediction.");
    return null;
  }

  try {
    // Convert blazeInput into a TensorFlow.js tensor
    const inputTensor = tf.tensor2d([blazeInput]);

    // Perform prediction
    const output = model.predict(inputTensor);

    // Apply softmax to get probabilities
    const probabilities = tf.softmax(output);
    const predictedClassTensor = probabilities.argMax(1);
    const predictedClass = (await predictedClassTensor.data())[0];

    // Clean up tensors to free memory
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