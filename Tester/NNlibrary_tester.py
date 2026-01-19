"""Simple tester"""
import matplotlib.pyplot as plt
import pandas as pd

from Dataset import Dataset as data
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelPrediction import ModelPrediction as pred
from ModelSaving import ModelSaving as ms

# 0.0. get some data (2025 15 min-power production in Italy)
dataset_freq = "15m"
ren_prod_italy = data.Dataset().getItalyEnergyProductionDataset(freq=dataset_freq)
# 0.1. Set the params
time_window = 96  # 48: 0.5 days | 96: 1 day | 192 : 2 days | 288: 3 days | 480: 5 days | 672: 7 days | 960: 10 days | 1920: 20 days
steps_ahead = 120
var_to_predict = "Hydro"  # 'Wind', 'Geothermal', 'Hydro', 'Photovoltaic', 'Biomass', 'Thermal', 'Self-consumption'
model_name = "hydro_prediction_1h"

# 0. Build the model
model = arch.ModelArch(modelStructure={"LSTM": [64, 64, 64, 64, 64], "FF": [500, 500]}).createRegressionModelArchitecture(dropout_FF=0.3)

# 1. Compile and train
trained_model = train.ModelTraining(model=model).trainModel(dataInDataFrameFormat=ren_prod_italy,
                                                            feature_variables=["year", "month", "day", "day_of_week", "hour", "minute"],   # "year" | "month" | "day" | "day_of_week" | "hour" | "minute"
                                                            target_variables=var_to_predict,
                                                            standardize=False,
                                                            split_method="seasonal-time-series",
                                                            time_window=time_window,
                                                            test_size=0.30,
                                                            batch_size=32,
                                                            validation_split=0.2,
                                                            epochs=100)

# 2. Evaluate the model
evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance()

# 3. Predict
prediction_dataset = pred.ModelPrediction(model=trained_model).predictTimeSeriesWithTrainedModel(dataInDataFrameFormat=ren_prod_italy,
                                                                                                 steps_ahead=steps_ahead,
                                                                                                 frequency=dataset_freq,
                                                                                                 date_column="index")

# 4. Save Model Weights
ms.ModelSaving(model=trained_model).saveModelWeights(save_dir="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models",
                                                     model_name=model_name)

# 4. Plot the prediction
plt.figure(figsize = (15, 5))
plt.plot(ren_prod_italy.sort_values(by="Date", ascending=True)[var_to_predict][-168:])
plt.plot(prediction_dataset[var_to_predict], color="red", linestyle="dashed")
plt.show()