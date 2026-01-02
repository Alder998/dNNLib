"""Class to evaluate the model performance"""

class ModelEvaluation:

    def __init__(self, model):
        self.model = model

    def evaluateModelPerformance(self):

        test_loss, test_acc = self.model["model"].evaluate(self.model["test_set"], self.model["test_labels"], verbose=2)

        print("INFO -- test loss: ", '{:,}'.format(test_loss))
        print("INFO -- test accuracy: ", '{:,}'.format(test_loss))

        return test_loss, test_acc
