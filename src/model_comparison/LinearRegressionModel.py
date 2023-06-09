from src.model_comparison.BaseModel import BaseModel


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


class LinearRegressionModel(BaseModel):
    def __init__(self):
        model = Pipeline([('regressor', Ridge())])
        param_dist = {
            'regressor__alpha': np.logspace(-3, 3, 7)
        }
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=7, cv=5, n_jobs=-1, random_state=42)
        super().__init__(random_search)
