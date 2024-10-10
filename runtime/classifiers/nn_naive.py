import numpy as np
from typing import Any
import pandas as pd
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, state
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

class NaiveNet():
    def __init__(self) -> None:
        self.layers = [
            Linear(20, 30), Tensor.relu,
            Linear(30, 1, bias=False), Tensor.sigmoid,
        ]    
        self.epochs = 1
        self.learning_rate = 0.1
        self.optimizer = Adam
        
    def __call__(self, x: Tensor) -> Any:
        return x.sequential(self.layers)

class NaiveModel():
    def __init__(self) -> None:
        self.model_class = NaiveNet
        
    def fit(self, X_in: np.ndarray, y) -> None:
        print("Fitting the model")
        self.model = self.model_class()
        X = Tensor(X_in)
        y = Tensor(y)
        
        optim = self.model.optimizer(get_parameters(self.model), lr=self.model.learning_rate)

        for epoch in range(self.model.epochs):
            # Forward
            y_pred = self.model(X)
            
            # Loss
            loss = (y_pred - y).pow(2).sum()
            
            # Backward
            optim.zero_grad()
            loss.backward()
            
            # Update
            optim.step()
            
            print(f"Epoch {epoch} Loss: {loss.item()}")
                
    def predict(self, X: pd.DataFrame):
        predictions = self.predict_proba(X)
        return [1 if x[0] >= 0.5 else 0 for x in predictions]
    
    def predict_proba(self, X_in: pd.DataFrame):
        # Return array of probabilities for each class (2)
        # Return a 2D array of shape (n_samples, n_classes)
        assert hasattr(self, "model"), "Model not trained"
        X = Tensor(X_in.to_numpy())
        predictions: Tensor = self.model(X)
        predictions = predictions.unsqueeze(1)
        predictions = predictions.sub(1, reverse=True)
        print(predictions.numpy())
        return predictions.numpy() 

