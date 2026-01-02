"""Simple tester"""

import numpy as np
import pandas as pd
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval

# 0.0. Creation of a stupid DataFrame of random data
random_dataset = pd.DataFrame(np.random.normal(size=(10000, 15))).set_axis([str(i) for i in np.arange(1, 16)],axis=1)

# 0. Build the model
model = arch.ModelArch(modelStructure={'FF': [500, 500]}).createRegressionModelArchitecture(dropout_FF=None)

# 1. Compile and train
trained_model = train.ModelTraining(model=model).trainModel(dataInDataFrameFormat=random_dataset,
                                                            feature_variables=["1", "2", "3", "12"],
                                                            target_variables="15",
                                                            test_size=0.30,
                                                            batch_size=4, validation_split=0.2, epochs=10)

# 2. Evaluate the model
evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance()