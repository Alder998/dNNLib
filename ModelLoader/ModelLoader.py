"""Class to load model from saved using model weights"""
import pandas as pd

from ModelArch import ModelArch as arch
from ModelPrediction import ModelPrediction as pred
from UtilsService import Plots as plts
import json
import numpy as np

class ModelLoader:

    def __init__(self):
        pass

    def predictTSWithloadedModel (self, modelPath, data, steps_ahead, frequency, date_column, target_variable):

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

        # 3. Create a new model Object
        modelObjForPred = {}
        modelObjForPred["var_to_predict"] = target_variable
        modelObjForPred["params"] = model_info["params"]
        modelObjForPred["time_window"] = model_info["time_window"]
        modelObjForPred["feature_scaler"] = model_info["feature_scaler"]
        modelObjForPred["target_scaler"] = model_info["target_scaler"]
        modelObjForPred["model"] = model

        # 4. Now, take care of data
        pred_hat, pred_upper, pred_lower = pred.ModelPrediction(model=modelObjForPred).predictTimeSeriesWithTrainedModel(dataInDataFrameFormat=data,
                                                                                                                          steps_ahead=steps_ahead,
                                                                                                                          frequency=frequency,
                                                                                                                          date_column=date_column,
                                                                                                                          confidence_area=True)

        # 5. Attempt of plotting
        plts.Plots().plotTimeSeriesPrediction(dataInDataFrameFormat=data, prediction_dataset=pred_hat, prediction_dataset_upper=pred_upper,
                                              prediction_dataset_lower=pred_lower, variable=target_variable, frequency=frequency,
                                              date_column=date_column, savePath=None)

        return pred_hat, pred_upper, pred_lower


    def predictGeoSpaceWithLoadedModel (self, modelPath, data, steps_ahead, frequency, date_column, target_variable):

        print("INFO - Loading Model and feeding it with saved Weights...")
        # 0. Open the model directory, loading model + weights + modelStructure
        with open(modelPath + "\\model_info.json", "r") as f:
            model_info = json.load(f)
        with open(modelPath + "\\model_structure.json", "r") as f:
            model_structure = json.load(f)

        # 1. Re-Create the model
        if model_info["problem"] == "regression":
            modelObj = arch.ModelArch(modelStructure=model_structure).createRegressionModelArchitecture(mode=model_info["mode"],
                                                                                                        adjacency_matrix=np.array(model_info["adjacency_matrix"]),
                                                                                                        input_shape=model_info["input_shape"],
                                                                                                        loss=model_info["loss_name"],
                                                                                                        peak_aware_loss_params=model_info["peak_aware_loss_params"])
        else:
            raise Exception("Problem " + str(model_info["problem"]) + " not implemented!")

        # 2. Isolate the model Object + put the trained weights inside the model
        model = modelObj["model"]
        model.load_weights(modelPath + "\\model_weights.weights.h5")

        # 3. Create a new model Object
        modelObjForPred = {}
        modelObjForPred["var_to_predict"] = target_variable
        modelObjForPred["params"] = model_info["params"]
        modelObjForPred["time_window"] = model_info["time_window"]
        modelObjForPred["feature_scaler"] = model_info["feature_scaler"]
        modelObjForPred["target_scaler"] = model_info["target_scaler"]
        modelObjForPred["space_variables_list"] = model_info["space_variables_list"]
        space_dataset = pd.read_csv(modelPath + "\\space_dataset.csv")
        modelObjForPred["space_variables"] = space_dataset[modelObjForPred["space_variables_list"]]
        modelObjForPred["modelStructure"] = model_structure
        modelObjForPred["model"] = model

        # 4. Now, take care of data
        pred_hat = pred.ModelPrediction(model=modelObjForPred).predictGeoSpatialWithTrainedModel(dataInDataFrameFormat=data,
                                                                                                  steps_ahead=steps_ahead,
                                                                                                  frequency=frequency,
                                                                                                  date_column=date_column)

        # 5. Attempt of plotting
        plts.Plots().plotGeospacePredictionFixedGrid(prediction_dataset=pred_hat,
                                                     variable=target_variable, date_column=date_column, savePath=None)

        return pred_hat