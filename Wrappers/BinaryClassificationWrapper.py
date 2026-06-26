"""
Wrapper to handle 2-class classification
"""

from Dataset import Dataset as dt
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval

class BinaryClassificationWrapper ():

    def __init__(self, modelStructure, target_variables, feature_variables):
        self.modelStructure = modelStructure
        self.target_variables = target_variables
        self.feature_variables = feature_variables
        pass

    def train_multiClass_model (self, data, epochs, test_size, split_method="random", standardize=False, batch_size=32,
                                validation_split=0.2, scaler="std"):

        # 0.0. Process the data
        data, categories_map = dt.Dataset().processDatasetForClassification(dataInDataFrameFormat=data,
                                                                            target_column=self.target_variables)
        # 1. Model Architecture
        model = arch.ModelArch(modelStructure=self.modelStructure).create2ClassificationModelArchitecture(mapping_classes=categories_map,
                                                                                                          mode="sequential")
        # 2. Model Training
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
                                                                    target_division=1,
                                                                    lag_series=[],
                                                                    scaler=scaler)
        # 3. Evaluate
        evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance(time_space=False)

        return model