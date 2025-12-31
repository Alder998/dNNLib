"""Simple tester"""

from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval

# 0. Build the model
model = arch.ModelArch(modelStructure={'FF': [500, 500]}).createModelArchitecture(dropout_FF=None)

# 1. Compile and train
trained_model = train.ModelTraining(model=model).trainModel(train_set=, train_labels=, batch_size=4, validation_split=0.2)

# 2. Evaluate the model
evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance(test_set=, test_labels=)