"""
Simple Multi-Layer Perceptron for simple regression problems
"""

from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval
from ModelSaving import ModelSaving as ms

class MLPWrapper:

    def __init__(self, modelStructure, feature_variables, target_variables):
        self.modelStructure = modelStructure
        self.feature_variables = feature_variables
        self.target_variables = target_variables
        pass

    def train_model(self, data, test_size, epochs, split_method="random", standardize=False, batch_size=32,
                    validation_split=0.2, target_division=1):

        # 0. Create Architecture
        model = arch.ModelArch(modelStructure=self.modelStructure).createRegressionModelArchitecture(mode="sequential")
        # 1. Train Model
        trained_model = train.ModelTraining(model=model).trainModel(dataInDataFrameFormat=data,
                                                                    feature_variables=self.feature_variables,
                                                                    target_variables=self.target_variables,
                                                                    standardize=standardize,
                                                                    split_method=split_method,
                                                                    seasonal_splits=None,
                                                                    time_window=1,
                                                                    test_size=test_size,
                                                                    batch_size=batch_size,
                                                                    validation_split=validation_split,
                                                                    epochs=epochs,
                                                                    target_division=target_division,
                                                                    lag_series=[])
        evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance(time_space=False)

        return model




