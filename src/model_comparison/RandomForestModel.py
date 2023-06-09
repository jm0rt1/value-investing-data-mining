from src.model_comparison.BaseModel import BaseModel


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


class RandomForestModel(BaseModel):
    def __init__(self):
        param_dist = {
            'n_estimators': np.arange(50, 310, 10),  # Updated range
            'max_depth': [None] + list(np.arange(5, 110, 5)),  # Updated range
            'min_samples_split': [2, 3, 5, 7, 10],  # Updated range
            'min_samples_leaf': [1, 2, 3, 4, 5],  # Updated range
            'max_features': [1.0, 'sqrt', 'log2'],  # Updated 'auto' to 1.0
            'bootstrap': [True, False]  # Added bootstrap
        }
        model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=200, cv=10, n_jobs=-1, random_state=42)  # Updated cv to 10
        super().__init__(random_search)
