from src.model_comparison.ml_models.BaseModel import BaseModel


from sklearn.ensemble import StackingRegressor


class StackingModel(BaseModel):
    def __init__(self, models):
        model = StackingRegressor(estimators=models)
        super().__init__(model)
