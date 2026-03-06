"""Class to load model from saved using model weights"""

from ModelArch import ModelArch as arch
import json

class ModelLoader:

    def __init__(self):
        pass

    def loadModel (self, modelPath):

        print("INFO - Loading Model and feeding it with saved Weights...")
        # 0. Open the model directory, loading model + weights + modelStructure
        with open(modelPath + "\\model_info.json", "r") as f:
            model_info = json.load(f)
        with open(modelPath + "\\model_structure.json", "r") as f:
            model_structure = json.load(f)

        # 1. Re-Create the model
        if model_info["problem"] == "regression":
            modelObj = arch.ModelArch(modelStructure=model_structure).createRegressionModelArchitecture(mode=model_info["mode"],
                                                                                                        adjacency_matrix=model_info["adjacency_matrix"],
                                                                                                        input_shape=model_info["input_shape"],
                                                                                                        loss=model_info["loss_name"],
                                                                                                        peak_aware_loss_params=model_info["peak_aware_loss_params"])
        else:
            raise Exception("Problem " + str(model_info["problem"]) + " not implemented!")

        # 2. Isolate the model Object + put the trained weights inside the model
        model = modelObj["model"]
        model.load_weights(modelPath + "\\model_weights.weights.h5")



        return 0