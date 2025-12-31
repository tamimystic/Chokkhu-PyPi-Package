from abc import ABC, abstractmethod


class BaseImageClassifier(ABC):
    def __init__(self):
        self.model = None
        self.history = None
        self.test_results = None

    @abstractmethod
    def Training(self, training_data: str, validation_data: str):
        pass

    @abstractmethod
    def Testing(self, testing_data: str):
        pass

    @abstractmethod
    def output(self):
        pass
