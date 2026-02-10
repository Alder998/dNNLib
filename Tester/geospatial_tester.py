"""Geospatial Model tester"""

from Dataset import Dataset as data
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelPrediction import ModelPrediction as pred
from ModelSaving import ModelSaving as ms
from VectorModule import VectorModule as vector

weatherData = data.Dataset().loadWeatherDataset()

modelStructure = {"GNN": {"layers": [32, 32, 32], "activation": "relu", "kernel_size":(3, 3), "strides": (1, 1), "padding": "same", "pool_size": (2, 2)},
                  "FF": {"layers": [500, 500], "activation": "relu"}}

# 0. Create Adjacency Matrix
adjacency_matrix = vector.VectorModule(modelStructure=modelStructure).createAdjacencyMatrixFromDataFrame(dataInDataFrameFormat=weatherData,
                                                                                                   target_variables="temperature",
                                                                                                   space_variables=["latitude", "longitude"])

# 1. Create Model
model = arch.ModelArch(modelStructure=modelStructure).createRegressionModelArchitecture(dropout_FF=0.2,
                                                                                        mode="functional",
                                                                                        adjacency_matrix=adjacency_matrix)

# 2. Train Model
trained_model = train.ModelTraining(model=model).trainGeospatialModel(dataInDataFrameFormat=weatherData,
                                                                      feature_variables=["year","month","day","hour"],
                                                                      space_variables=["latitude", "longitude"],
                                                                      target_variables="temperature",
                                                                      standardize=False,
                                                                      split_method="seasonal-time-series",
                                                                      seasonal_splits=12,
                                                                      time_window=24,
                                                                      test_size=0.30,
                                                                      batch_size=32,
                                                                      validation_split=0.2,
                                                                      epochs=10)
# 2. Evaluate Model
evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance()

# 3. Predict the future
prediction_dataset = pred.ModelPrediction(model=trained_model).predictGeoSpatialWithTrainedModel(dataInDataFrameFormat=weatherData,
                                                                                                 steps_ahead=20,
                                                                                                 frequency="1h",
                                                                                                 date_column="date")