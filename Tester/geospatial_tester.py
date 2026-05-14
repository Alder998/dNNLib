"""Geospatial Model tester"""

from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelPrediction import ModelPrediction as pred
from ModelSaving import ModelSaving as ms
from VectorModule import VectorModule as vector
from UtilsService import Plots as plot
import pandas as pd

weatherData = pd.read_csv(r"C:\Users\alder\Downloads\3m_weather.csv")

# 0.0. Set the input
modelStructure = {"GConv": {"layers": [32], "activation": "relu"},
                  "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.0},
                  "FF": {"layers": [200, 200], "activation": "relu"}}
space_variables = ["latitude", "longitude"]
feature_variables = ["month","day","hour"]   # "year","month","day","hour"
time_window = 96
target_variables = "temperature"  #'temperature' | 'precipitation' | 'windSpeed' | 'humidity_mean' | 'cloudCover' | 'pressure_msl'
steps_ahead=20
date_column="date"

# 0. Create Adjacency Matrix
adjacency_matrix = vector.VectorModule(modelStructure=modelStructure).createAdjacencyMatrixFromDataFrame(dataInDataFrameFormat=weatherData,
                                                                                                         target_variables=target_variables,
                                                                                                         space_variables=space_variables,
                                                                                                         sigma=None)


# 1. Create Model
model = arch.ModelArch(modelStructure=modelStructure).createRegressionModelArchitecture(mode="functional",
                                                                                        adjacency_matrix=adjacency_matrix,
                                                                                        input_shape=(time_window, adjacency_matrix.shape[0], len(feature_variables)))

# 2. Train Model
trained_model = train.ModelTraining(model=model).trainGeospatialModel(dataInDataFrameFormat=weatherData,
                                                                      feature_variables=feature_variables,
                                                                      space_variables=space_variables,
                                                                      target_variables=target_variables,
                                                                      standardize=False,
                                                                      split_method="time-series",
                                                                      seasonal_splits=12,
                                                                      time_window=time_window,
                                                                      test_size=0.30,
                                                                      batch_size=32,
                                                                      validation_split=0.2,
                                                                      epochs=200)
# 2. Evaluate Model
evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance(time_space=True)

# 3. Predict the future
prediction_dataset = pred.ModelPrediction(model=trained_model).predictGeoSpatialWithTrainedModel(dataInDataFrameFormat=weatherData,
                                                                                                 steps_ahead=steps_ahead,
                                                                                                 frequency="1h",
                                                                                                 date_column=date_column)

prediction_dataset.to_excel(r"C:\Users\alder\Downloads\first_geospace.xlsx", index=False)

# 4. Save Model Weights
ms.ModelSaving(model=trained_model).saveModelWeights(save_dir="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models",
                                                     model_name="geospace_mixed_model")

# 5. Plot prediction
plot.Plots().plotGeospacePredictionFixedGrid(dataInDataFrameFormat=weatherData,
                                             prediction_dataset=prediction_dataset,
                                             variable=target_variables,
                                             date_column=date_column,
                                             colorScale="rainbow",
                                             savePath="C:\\Users\\alder\\Downloads\\prediction_heatmap_" + target_variables + ".gif")