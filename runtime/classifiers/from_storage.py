import os
from pathlib import Path
import pickle
from typing import Dict
import pandas as pd

from runtime.analysis import Analysis, Filter
from runtime.model import Model

class ModelFromStorage():
    def __init__(self, filter: Filter, path: Path):
        self.skip_resample = True
        # Load model paths into a dictionary where the key is the fold number
        self.models: Dict[int, Path] = {}
        self.filter: Filter = filter
        self.analysis = Analysis(filter=self.filter, records_path=path)
        self.results = self.analysis.save_analysis_db()
        self.current_model = None
        self.from_storage = True
        
        # Get range of folds
        folds = self.results["fold"].unique()
        assert len(folds) == len(self.results), "We selected more than one combination of algorithms"
    
        print(self.analysis.analytics_file)

        # Check that all paths are valid
        for index, row in self.results.iterrows():
            fold = row["fold"]
            path = Path(row["model_file"])
            assert os.path.exists(path), f"Path {path} does not exist"
            assert fold not in self.models, f"Fold {fold} already exists in models"
            assert isinstance(fold, int), "Fold is not an integer"
            self.models[fold] = path

    def fit(self, X, y, fold):
        # Load pickled model
        print("Prefitted model, skipping fit, loading from storage.")
        self.current_model = pickle.load(open(self.models[fold], "rb"))    
        assert isinstance(self.current_model, Model), "Model is not the right type of object. It needs to be wrapped in a runtime Model object."    

    def predict(self, X: pd.DataFrame):
        return self.current_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        if hasattr(self.current_model, "predict_proba"):
            return self.current_model.predict_proba(X)
        return self.current_model.predict(X)
