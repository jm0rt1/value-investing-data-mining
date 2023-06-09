from src.model_comparison.BaseModel import BaseModel


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


class SupportVectorMachineModel(BaseModel):
    def __init__(self):
        param_grid = {
            'C': np.logspace(-3, 3, 7),
            'epsilon': np.logspace(-3, 3, 7),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        model = SVR()
        grid_search = GridSearchCV(
            model, param_grid, cv=5, n_jobs=-1)
        super().__init__(grid_search)
