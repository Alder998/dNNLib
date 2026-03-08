"""Tester for model loading"""

from Dataset import Dataset as data
from ModelLoader import ModelLoader as loader
from datetime import datetime
import pandas as pd

# 0. Data
data = data.Dataset().getItalyEnergyProductionDataset(freq="15min")
data = data[data.index <= datetime(2025, 7, 1)]

# 1. Stored Model
pred, pred_upper, pred_lower = loader.ModelLoader().predictTSWithloadedModel(modelPath="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models\\thermal_prediction_15min",
                                                                             data=data,
                                                                             target_variable="Thermal",
                                                                             steps_ahead=96,
                                                                             frequency="15min",
                                                                             date_column="index")