import abc

class method:
    
    method_name = None
    method_description = None
    
    data = None
    
    method_start_time = None
    method_stop_time = None
    method_running_time = None
    method_training_time = None
    method_testing_time = None

    def __init__(self, mName=None, mDescription=None):
        self.methodName = mName
        self.method_description = mDescription

    @abc.abstractmethod
    def run(self, trainData, trainLabel, testData):
        return
