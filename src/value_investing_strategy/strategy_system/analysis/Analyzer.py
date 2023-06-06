

import abc


class Analyzer(abc.ABC):
    def __init__(self):
        self.uid = None

    def run(self, ):
        pass

    def save(self, ):
        pass

    def to_csv(self, ):
        pass
