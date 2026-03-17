"""Class to save the model weights for reusability"""

import os
import json

class ModelSaving:

    def __init__(self, model):
        self.model = model
        pass

    def saveModelWeights(self, save_dir, model_name="model"):

        if save_dir is not None:
            custom_directory = save_dir + "\\" + model_name

            # 0. Create directory, if not existing
            if not os.path.exists(custom_directory):
                os.mkdir(custom_directory)

            # 1. Save Model Weights into the directory
            self.model["model"].save_weights(custom_directory + "\\model_weights.weights.h5")

            # 2. Save the Model Structure to re-train the model
            with open(custom_directory + "\\model_structure.json", "w") as f:
                json.dump(self.model["modelStructure"], f)

            # 3. Create the model info to store other useful object for model re-training
            model_info={}
            model_info["problem"] = self.model["problem"]
            model_info["input_shape"] = self.model["input_shape"]
            model_info["loss_name"] = self.model["loss_name"]
            model_info["peak_aware_loss_params"] = self.model["peak_aware_loss_params"]
            model_info["mode"] = self.model["mode"]
            model_info["adjacency_matrix"] = self.model["adjacency_matrix"].tolist()
            model_info["time_window"] = self.model["time_window"]
            model_info["params"] = self.model["params"]
            model_info["var_to_predict"] = self.model["var_to_predict"]
            model_info["feature_scaler"] = self.model["feature_scaler"]
            model_info["target_scaler"] = self.model["target_scaler"]
            # Space variables must be added only for geo-space Model
            if "space_variables" in self.model.keys():
                self.model["space_variables"].to_csv(custom_directory + "\\space_dataset.csv")
                model_info["space_variables_list"] = self.model["space_variables_list"]

            with open(custom_directory + "\\model_info.json", "w") as f:
                json.dump(model_info, f)

            print("INFO - Model weights, structure, info saved correctly")