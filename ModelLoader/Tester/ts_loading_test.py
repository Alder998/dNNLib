"""Tester for model loading"""

from ModelLoader import ModelLoader as loader
import pandas as pd

# 0. Data
data = pd.read_excel("C:\\Users\\alder\\Downloads\\aggregate_load_by_bidding_zone.xlsx")
data = data[data["Bidding_zone"]=="IT-NORD"].sort_values(by="Date", ascending=True).reset_index(drop=True)

# 1. Stored Model
pred, pred_upper, pred_lower = loader.ModelLoader().predictTSWithloadedModel(modelPath="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models\\thermal_prediction_15min",
                                                                             data=data,
                                                                             target_variable="Actual Load",
                                                                             steps_ahead=96,
                                                                             frequency="15min",
                                                                             date_column="Date")