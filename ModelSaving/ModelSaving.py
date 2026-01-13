"""Class to save the model weights for reusability"""

import os

class ModelSaving:

    def __init__(self, model):
        self.model = model
        pass

    def saveModelWeights(self, save_dir, model_name="model"):

        custom_directory = save_dir + "\\" + model_name
        # 0. Create directory, if not existing
        if not os.path.exists(custom_directory):
            os.mkdir(custom_directory)

        # 1. Save Model Weights into the directory
        self.model["model"].save_weights(custom_directory + "\\model_weights.weights.h5")
        print("INFO - Model weights saved correctly")
