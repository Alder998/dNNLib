"""Class to ease the Time Series Predictions"""

import matplotlib.pyplot as plt
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelPrediction import ModelPrediction as pred
from ModelSaving import ModelSaving as ms

class TimeSeriesWrapper:

    def __init__(self, modelStructure, feature_variables, time_window, target_variables, date_column, frequency):
        self.modelStructure = modelStructure
        self.feature_variables = feature_variables
        self.time_window = time_window
        self.target_variables = target_variables
        self.date_column = date_column
        self.frequency = frequency

    def trainPredictAndSaveTimeSeriesModel (self, data, prediction_steps_ahead, epochs, loss="MSE",
                                            peak_aware_loss_params={"alpha": 3.0, "beta": 2.0,"gamma": 1.0, "delta": 2.0, "peak_threshold": 0.8},
                                            test_size=0.30, validation_split=0.2, dropout_FF=0, standardize=False,
                                            split_method="time-series", seasonal_splits=12, batch_size=32, save_dir=None,
                                            model_save_name="model", plot=False, plot_save_dir=None):

        # 0. Build the model
        model = arch.ModelArch(modelStructure=self.modelStructure).createRegressionModelArchitecture(mode="functional",
                                                                                                      dropout_FF=dropout_FF,
                                                                                                      input_shape=(self.time_window, len(self.feature_variables)),
                                                                                                      loss=loss,   # "peak-aware-MSE" | "MSE"
                                                                                                      peak_aware_loss_params=peak_aware_loss_params)
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
                                                                    epochs=epochs)

        # 2. Evaluate the model
        evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance()

        # 3. Predict
        prediction_dataset = pred.ModelPrediction(model=trained_model).predictTimeSeriesWithTrainedModel(dataInDataFrameFormat=data,
                                                                                                         steps_ahead=prediction_steps_ahead,
                                                                                                         frequency=self.frequency,
                                                                                                         date_column=self.date_column)

        # 4. Save Model Weights
        ms.ModelSaving(model=trained_model).saveModelWeights(save_dir=save_dir,
                                                             model_name=model_save_name)

        # 4. Plot the prediction
        if plot:
            plt.figure(figsize = (15, 5))
            plt.plot(data.sort_values(by="Date", ascending=True)[self.target_variables][(-672 if self.frequency=="15min" else -168):])
            plt.plot(prediction_dataset[self.target_variables], color="red", linestyle="dashed")
            plt.show()