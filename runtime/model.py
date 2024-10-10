from typing import Callable, Dict, Optional, Self, Any
from skopt import BayesSearchCV
from skopt.space import Dimension
import numpy as np
from sklearn.base import BaseEstimator
from runtime.evaluation import accuracy


type SearchSpace = Dict[str, Dimension]
type ModelDict = dict[str, dict[str, Any]] 



"""EXAMPLES

EXAMPLE_SEARCH_SPACE: SearchSpace = {
    "n_estimators": Integer(100, 1000),
    "max_depth": Integer(1, 10),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["auto", "sqrt", "log2"]),
    "bootstrap": Categorical([True, False]),
}
EXAMPLE_MODEL_LIST = {
    "RandomForestClassifier": {
        "estimator": RandomForestClassifier,
        "search_spaces": EXAMPLE_SEARCH_SPACE,
    }
}
"""

class Model:
    """
    A model to be used in the training process.

    This is a wrapper around a scikit-learn estimator, and it contains the search space for the estimator's hyperparameters.
    It exists to make the training process more convenient, and to allow for serialization and deserialization of the model.
    
    Parameters:
    name (str): The name of the model (labelling purposes usually).
    estimator (BaseEstimator): The scikit-learn estimator to be used in the training process.
    search_spaces (SearchSpace): The search space for the estimator's hyperparameters.
    """
    def __init__(self, name: str, 
                 estimator: BaseEstimator, 
                 search_spaces: Optional[SearchSpace] = None, 
                 scorer: Optional[Callable] = None,
                 skip_resample: bool = False,
                 **kwargs) -> None:
        self.estimator: BaseEstimator = estimator
        self.search_spaces = search_spaces
        self.name: str = name
        self.scorer: Callable = scorer if scorer else accuracy
        self.skip_resample: bool = skip_resample
        self.loaded_from_storage: bool = False

        if hasattr(self.estimator, "skip_resample"):
            self.skip_resample = self.estimator.skip_resample
        if hasattr(self.estimator, "loaded_from_storage"):
            self.loaded_from_storage = self.estimator.loaded_from_storage

        if self.search_spaces is not None and not self.loaded_from_storage:
            self.model = BayesSearchCV(estimator=self.estimator, 
                                    search_spaces=self.search_spaces,
                                    scoring=scorer, 
                                    refit=True,
                                    cv=5,
                                    n_jobs=5,
                                    n_iter=10,
                                    **kwargs)
        else:
            self.model = self.estimator
            self.model.scorer_ = self.scorer # type: ignore
            
        assert hasattr(self.model, "fit") and callable(self.model.fit), "Model must have a fit method" # type: ignore
        assert hasattr(self.model, "predict") and callable(self.model.predict), "Model must have a predict method" # type: ignore


    def train(self, X, y, **kwargs) -> None:
        np.int = np.int64 # type: ignore # https://github.com/scikit-optimize/scikit-optimize/issues/1171#issuecomment-1584561856
        self.model.fit(X, y, **kwargs)# type: ignore


    def predict(self, X) -> np.ndarray:
        np.int = np.int64  # type: ignore # https://github.com/scikit-optimize/scikit-optimize/issues/1171#issuecomment-1584561856
        return self.model.predict(X) # type: ignore
    
    def predict_proba(self, X) -> np.ndarray:
        np.int = np.int64
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        # Return the same shape as the predict_proba method of the RandomForestClassifier
        predictions = self.model.predict(X)
        return np.array([[1 - p, p] for p in predictions])


    def score(self, X, y) -> float:
        # TODO - Could be updated to use a instance of a `Scorer` class? This would mean we can just pass a scorer in similarly to `Resampler`
        # I don't think this is necessary in retrospect but it is possible.

        # Note the scorer can be a string or a callable. 
        # If it is a callable, it is assumed to be a function that takes the model, X, and y as parameters.
        if type(self.scorer) is not str:
            return self.scorer(self.model, X, y) # type: ignore
        return self.model.score(X, y) # type: ignore
    

    def __str__(self) -> str:
        return self.name
    

    def __repr__(self) -> str:
        """
        Returns a string representation of the model that can be used to reconstruct it later.
        This is useful to store in the results file metadata to allow for easy reconstruction of the model.
        """
        return str(self.__dict__)
    

    @classmethod
    def from_json(cls, name:str, json: ModelDict, disable_bayes_search: bool = False) -> Self:
        # Standardize the JSON

        if "search_spaces" not in json or disable_bayes_search:
            # Ensure that the search spaces are not used if they are not provided but they are defined.
            json["search_spaces"] = None # type: ignore

        if "scorer" not in json:
            json["scorer"] = None # type: ignore

        assert "estimator" in json, "Model JSON must contain an estimator"
        assert hasattr(json["estimator"], "fit"), "Estimator must have a fit method"
        assert callable(json["estimator"].fit), "Estimator must have a fit method" # type: ignore
        assert hasattr(json["estimator"], "predict"), "Estimator must have a predict method"
        assert callable(json["estimator"].predict), "Estimator must have a predict method" # type: ignore

        if "skip_resample" not in json:
            json["skip_resample"] = False # type: ignore
        
        assert type(json["skip_resample"]) == bool, "skip_resample must be a boolean"
#        assert type(json["estimator"]) == BaseEstimator, "estimator must be a scikit-learn estimator"
#        assert callable(json["scorer"]), "Estimator must have a fit method"
#        assert type(json["search_spaces"]) == SearchSpace or json["search_spaces"] is None, "search_spaces must be a dictionary of skopt.space.Dimension objects or None"
        
        return cls(name, json["estimator"], json["search_spaces"], json["scorer"], skip_resample=json["skip_resample"]) # type: ignore
    

class TestModel:
    def test_serde(self):
        # TODO - Add test for serialization and deserialization
        assert True
