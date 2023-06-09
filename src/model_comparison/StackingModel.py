from src.model_comparison.BaseModel import BaseModel


from sklearn.ensemble import StackingRegressor


class StackingModel(BaseModel):
    def __init__(self, models):
        model = StackingRegressor(estimators=models)
        super().__init__(model)
