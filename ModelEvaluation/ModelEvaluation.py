"""Class to evaluate the model performance"""

class ModelEvaluation:

    def __init__(self, model):
        self.model = model

    def evaluateModelPerformance(self, time_space=False):

        if time_space:
            test_loss, test_acc = self.model["model"].evaluate(self.model["test_set"].transpose(0, 1, 3, 2), self.model["test_labels"], verbose=2)
        else:
            test_loss, test_acc = self.model["model"].evaluate(self.model["test_set"], self.model["test_labels"], verbose=2)

        print("INFO -- test loss: ", '{:,}'.format(test_loss))
        if "classification" in self.model["problem"]:
            print("INFO -- test accuracy: ", '{:,}'.format(round(test_acc * 100, 2)) + " %")

        return test_loss, test_acc
