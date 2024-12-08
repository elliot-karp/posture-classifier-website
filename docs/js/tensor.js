// Load the model and make the promise accessible globally
export const modelPromise = loadModel();

async function loadModel() {
  try {
    console.log("Loading TensorFlow.js model...");
    const model = await tf.loadLayersModel("tfjs_model/model.json");
    console.log("Model loaded successfully:", model);
    return model;
  } catch (error) {
    console.error("Error loading the model:", error);
    return null;
  }
}

export async function queryModel(blazeInput) {
  const model = await modelPromise; // Now modelPromise is properly exported and accessible
  if (!model) {
    console.error("Model not loaded, skipping prediction.");
    return null;
  }

  try {
    const inputTensor = tf.tensor2d([blazeInput]);
    //console.log("Input Tensor for Model:", inputTensor.toString());

    const output = model.predict(inputTensor);
    const probabilities = output.softmax();
    const predictedClassTensor = probabilities.argMax(1);
    const predictedClass = (await predictedClassTensor.data())[0];
    //console.log("Predicted Class:", predictedClass);

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