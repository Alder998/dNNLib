"""Geospatial Model tester"""

from Dataset import Dataset as data
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelPrediction import ModelPrediction as pred
from ModelSaving import ModelSaving as ms
from VectorModule import VectorModule as vector

weatherData = data.Dataset().loadWeatherDataset()

# 0.0. Set the input
modelStructure = {"GConv": {"layers": [32, 32, 32], "activation": "relu"},
                  "LSTM": {"layers": [64, 64, 64, 64, 64], "activation": "tanh", "dropout": 0.0},
                  "FF": {"layers": [500, 500], "activation": "relu"}}
space_variables = ["latitude", "longitude"]
feature_variables = ["day","hour"]   # "year","month","day","hour"
time_window = 96
target_variables = "temperature"
steps_ahead=20
date_column="date"

# 0. Create Adjacency Matrix
adjacency_matrix = vector.VectorModule(modelStructure=modelStructure).createAdjacencyMatrixFromDataFrame(dataInDataFrameFormat=weatherData,
                                                                                                   target_variables=target_variables,
                                                                                                   space_variables=space_variables)

# 1. Create Model
model = arch.ModelArch(modelStructure=modelStructure).createRegressionModelArchitecture(dropout_FF=0.2,
                                                                                        mode="functional",
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

print(prediction_dataset)
prediction_dataset.to_excel(r"C:\Users\alder\Downloads\first_geospace.xlsx")