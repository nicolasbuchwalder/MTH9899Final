import pickle
from sklearn.metrics import r2_score
import pandas as pd
import lightgbm

class Predictor:
    """
    Class to load the model and predict and evaluate predictions
    """
    # getting model
    def __init__(self, mod):
        self._mod = mod
        with open((self._mod / "lgb_model1.pkl").resolve(), "rb") as file: 
            self._model = pickle.load(file)
    # predicting 
    def predict(self, X):
        X['Market Regime'] = X['Market Regime'].astype('category')
        return self._model.predict(X) / 1e4
    # getting r_squared
    def evaluate(self, y_pred, y, sample_weights):
        print(f"Weighted R2 is {r2_score(y, y_pred, sample_weight=sample_weights)}")

