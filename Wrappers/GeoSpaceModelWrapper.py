"""Wrapper to make easy and fast geospace Predictions"""

from Dataset import Dataset as dt
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelPrediction import ModelPrediction as pred
from ModelSaving import ModelSaving as ms
from VectorModule import VectorModule as vector
from UtilsService import Plots as plots

class GeoSpaceModelWrapper:

    def __init__(self, modelStructure, space_variables, feature_variables, time_window, target_variables, date_column, frequency, lags):
        self.modelStructure = modelStructure
        self.space_variables = space_variables
        self.feature_variables = feature_variables
        self.time_window = time_window
        self.target_variables = target_variables
        self.date_column = date_column
        self.frequency = frequency
        self.lags = lags
        pass

    def trainPredictAndSaveGeospaceModel (self, data, prediction_steps_ahead, epochs, test_size=0.30, validation_split=0.2,
                                          sigma_adjacency=None, standardize=False, split_method="time-series",
                                          seasonal_splits=12, batch_size=32, save_dir=None, model_save_name="model", plot=False,
                                          plot_save_dir=None, target_division=1, date_column_format="%Y-%m-%d %H:%M:%S"):

        # 0.0. Process the data
        data = dt.Dataset().processDatasetForTimeSeries(dataInDataFrameFormat=data,
                                                        date_column=self.date_column,
                                                        target_column=self.target_variables,
                                                        date_column_format=date_column_format,
                                                        frequency=self.frequency,
                                                        lag_series=self.lags)

        # 0. Create Adjacency Matrix
        adjacency_matrix = vector.VectorModule(modelStructure=self.modelStructure).createAdjacencyMatrixFromDataFrame(dataInDataFrameFormat=data,
                                                                                                                      target_variables=self.target_variables,
                                                                                                                      space_variables=self.space_variables,
                                                                                                                      sigma=sigma_adjacency)

        # 1. Create Model
        model = arch.ModelArch(modelStructure=self.modelStructure).createRegressionModelArchitecture(mode="functional",
                                                                                                    adjacency_matrix=adjacency_matrix,
                                                                                                    input_shape=(self.time_window,
                                                                                                                 adjacency_matrix.shape[0],
                                                                                                                 len(self.feature_variables)))

        # 2. Train Model
        trained_model = train.ModelTraining(model=model).trainGeospatialModel(dataInDataFrameFormat=data,
                                                                              feature_variables=self.feature_variables,
                                                                              space_variables=self.space_variables,
                                                                              target_variables=self.target_variables,
                                                                              standardize=standardize,
                                                                              split_method=split_method,
                                                                              seasonal_splits=seasonal_splits,
                                                                              time_window=self.time_window,
                                                                              test_size=test_size,
                                                                              batch_size=batch_size,
                                                                              validation_split=validation_split,
                                                                              epochs=epochs,
                                                                              target_division=target_division,
                                                                              lag_series=self.lags)
        # 2. Evaluate Model
        evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance(time_space=True)

        # 3. Predict the future
        prediction_dataset = pred.ModelPrediction(model=trained_model).predictGeoSpatialWithTrainedModel(dataInDataFrameFormat=data,
                                                                                                         steps_ahead=prediction_steps_ahead,
                                                                                                         frequency=self.frequency,
                                                                                                         date_column=self.date_column,
                                                                                                         target_division=target_division)
        # 4. Save Model Weights
        ms.ModelSaving(model=trained_model).saveModelWeights(
                       save_dir=save_dir,
                       model_name=model_save_name)

        # 5. Plot prediction
        if plot:
            plots.Plots().plotGeospacePredictionFixedGrid(prediction_dataset=prediction_dataset,
                                                         variable=self.target_variables,
                                                         date_column=self.date_column,
                                                         colorScale="rainbow",
                                                         savePath=plot_save_dir)