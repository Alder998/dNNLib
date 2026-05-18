"""Class to ease the Time Series Predictions"""

from Dataset import Dataset as dt
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelPrediction import ModelPrediction as pred
from ModelSaving import ModelSaving as ms
from UtilsService import Plots as plts

class TimeSeriesWrapper:

    def __init__(self, modelStructure, feature_variables, time_window, target_variables, date_column, frequency, lags):
        self.modelStructure = modelStructure
        self.feature_variables = feature_variables
        self.time_window = time_window
        self.target_variables = target_variables
        self.date_column = date_column
        self.frequency = frequency
        self.lags = lags

    def trainPredictAndSaveTimeSeriesModel (self, data, prediction_steps_ahead, epochs, loss="MSE",
                                            peak_aware_loss_params={"alpha": 3.0, "beta": 2.0,"gamma": 1.0, "delta": 2.0, "peak_threshold": 0.8},
                                            test_size=0.30, validation_split=0.2, standardize=False,
                                            split_method="time-series", seasonal_splits=12, batch_size=32, save_dir=None,
                                            model_save_name="model", plot=False, plot_save_dir=None, confidence_area=True,
                                            target_division=1, date_column_format="%Y-%m-%d %H:%M:%S"):

        # 0.0. Process the data
        data = dt.Dataset().processDatasetForTimeSeries(dataInDataFrameFormat=data,
                                                        date_column=self.date_column,
                                                        target_column=self.target_variables,
                                                        date_column_format=date_column_format,
                                                        frequency=self.frequency,
                                                        lag_series=self.lags)

        # 0. Build the model
        model = arch.ModelArch(modelStructure=self.modelStructure).createRegressionModelArchitecture(mode="functional",
                                                                                                      input_shape=(self.time_window, len(self.feature_variables) + len(self.lags)),
                                                                                                      loss=loss,   # "peak-aware-MSE" | "MSE"
                                                                                                      peak_aware_loss_params=peak_aware_loss_params,
                                                                                                      target_variables=len(self.target_variables))
        # 1. Compile and train
        trained_model = train.ModelTraining(model=model).trainModel(dataInDataFrameFormat=data,
                                                                    feature_variables=self.feature_variables,
                                                                    target_variables=self.target_variables,
                                                                    standardize=standardize,
                                                                    split_method=split_method,  # "time-series" | "seasonal-time-series" | "random"
                                                                    seasonal_splits=seasonal_splits,
                                                                    time_window=self.time_window,
                                                                    test_size=test_size,
                                                                    batch_size=batch_size,
                                                                    validation_split=validation_split,
                                                                    epochs=epochs,
                                                                    target_division=target_division,
                                                                    lag_series=self.lags)

        # 2. Evaluate the model
        evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance()

        # 3. Predict
        prediction_dataset, upper_95, lower_95 = pred.ModelPrediction(model=trained_model).predictTimeSeriesWithTrainedModel(dataInDataFrameFormat=data,
                                                                                                         steps_ahead=prediction_steps_ahead,
                                                                                                         frequency=self.frequency,
                                                                                                         date_column=self.date_column,
                                                                                                         confidence_area=confidence_area,
                                                                                                         target_division=target_division,
                                                                                                         lag_series=self.lags)

        # 4. Save Model Weights
        ms.ModelSaving(model=trained_model).saveModelWeights(save_dir=save_dir,
                                                             model_name=model_save_name)

        # 4. Plot the prediction
        if plot:
            plts.Plots().plotTimeSeriesPrediction(dataInDataFrameFormat=data, prediction_dataset=prediction_dataset,
                                                  prediction_dataset_upper=upper_95,
                                                  prediction_dataset_lower=lower_95, variable=self.target_variables,
                                                  frequency=self.frequency,
                                                  date_column=self.date_column, savePath=plot_save_dir,
                                                  target_division=target_division)