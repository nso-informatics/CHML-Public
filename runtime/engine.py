"""
    This file contains the engine of the project.
    It is what runs the models and the training, recording the entire process for later evaluation.
"""

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import Dict, List, Tuple
import multiprocessing as mp
import setproctitle
from pathlib import Path
import sys

from runtime.feature_engineering import create_powers, create_ratios, primary_weight

# directory reach
directory = Path(__file__)

# setting path
sys.path.append(directory.parent.parent.absolute().__str__())

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from runtime.evaluation import *
from runtime.model import Model, ModelDict
from runtime.resampling import *
from runtime.storage.file import FileStorage
from runtime.storage.db import DBStorage
from runtime.storage.storage import BaseStorage


class Engine:
    """
    The Engine.

    It supports multiple models and resamplers, and can run the training process in parallel using multiple processes.
    The results of the training are saved to a file in the HDF5 format. Models are simple and only need to be able to
    support the `fit` and `predict` methods. Resamplers are also simple and only need to be able to support the
    `fit_resample` method.

    Parameters:
    models (List[Model] | ModelDict): The models to be used in the training.
    resamplers (List[Resampler]): The resamplers to be used in the training.
    scorers (Dict[str, str | Callable]): The scorers to be used in the training.
    data_file (Path): The path to the data input file. TODO - Add loading?
    X (DataFrame): The data to be used in the training.
    y (np.ndarray): The labels to be used in the training.
    verbosity (int): The verbosity level of the engine. 0 = silent, 1 = results.h5, 2 = verbose.
    records_dir (Path): The directory where the results.h5 file will be saved.
    max_workers (int): The maximum number of processes to be used in the training.
    output_name (str): The name of the output file `results`.
    output_format (str): The format of the output file. Can be `h5`, `json`, `csv`, or `pickle`(DataFrame).
    overwrite (bool): Whether to overwrite the output file if it already exists.
    disable_bayes_search (bool): Whether to disable the use of BayesSearchCV for hyperparameter optimization.
    kwargs: Additional keyword arguments.
    """

    output_lock = multiprocessing.Lock()
    log_lock = multiprocessing.Lock()

    def __init__(
        self,
        models: List[Model] | ModelDict,
        resamplers: ResamplerList,
        scorers: List[Metric],
        tag: str = "no_tag",
        X: pd.DataFrame = pd.DataFrame(),
        y: pd.Series = pd.Series(),
        verbosity: int = 1,
        records_dir: Path = Path("./records"),
        max_workers: int = 1,
        output_name: str = "results",
        output_format: str = "csv",
        disable_bayes_search: bool = False,
        cross_validate: bool = True,
        save_probabilities: bool = True,
        use_database: bool = True,
        use_optimal_features: bool = False,
        use_synthetic_ratios: bool = False,
        use_synthetic_powers: bool = False,
        use_primary_weights: bool = False,
        cross_validators: List[BaseCrossValidator] = [StratifiedKFold(n_splits=5, random_state=None)],
        layered_models: List[Model] = [],
        passthrough_precentage: Optional[float] = None,
        extra_resampled_train_files: Optional[List[Tuple[str, str]]] = None,
    ):

        # Handle data configuration tasks
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.empty or len(y) == 0:
            raise ValueError("Data must be provided.")

        self.X = X
        self.y = y
        self.verbosity = verbosity
        self.records_dir = records_dir
        self.max_workers = max_workers
        self.output_name = output_name
        self.output_format = output_format
        self.cross_validators = cross_validators
        self.cross_validate = cross_validate
        self.resamplers: List[Resampler] = []
        self.tag = tag
        self.save_probabilities = save_probabilities
        self.log_file = self.records_dir / "log.txt"
        self.Storage = DBStorage if use_database else FileStorage
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.use_optimal_features = use_optimal_features
        self.use_synthetic_ratios = use_synthetic_ratios
        self.use_synthetic_powers = use_synthetic_powers
        self.use_primary_weights = use_primary_weights
        self.layered_models = layered_models
        self.pass_through_percentage = passthrough_precentage
        self.extra_resampled_train_files = extra_resampled_train_files

        try:
            # Handle model configuration tasks and creation of models Objects.
            self._deserialize_models(models, scorers, disable_bayes_search=disable_bayes_search)
            # Handle loading of resamplers into Resampler objects
            self._deserialize_resamplers(resamplers)

        except Exception as e:
            self._log(f"Error while setting up engine: {e}")
            raise e

    def run(self) -> None:
        """
        Initiates the training process for all combinations of models and resamplers.

        This method creates a ProcessPoolExecutor with a number of workers equal to `max_workers`.
        It then submits tasks to this executor for each combination of model and resampler.
        Each task is a call to the `_train` method with a specific model and resampler.

        Returns:
        self: The engine instance.
        """
        # Dispatch the training tasks.
        executor = ProcessPoolExecutor(mp_context=mp.get_context("fork"), max_workers=self.max_workers)
        futures = []
        for cross_validator in self.cross_validators:
            for resampler in self.resamplers:
                for model in self.models:
                    if self.max_workers == 1:
                        self._train(model, resampler, cross_validator)
                    else:
                        futures.append(executor.submit(self._train, model, resampler, cross_validator))

        # Wait for tasks to finish
        # TODO Monitor progress and ensure progress is being made.
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                self._log(f"Error while training. {self.tag}: {e}")
                raise e

        executor.shutdown()
        self._log(f"Done Training {self.tag}.")

    def telegram(self, msg: str):
        try:
            chat_id = "6556340412"
            token = os.environ.get("TELEGRAM_API_TOKEN")
            url = f"https://api.telegram.org/bot{token}"
            params = {"chat_id": chat_id, "text": msg}
            r = requests.get(url + "/sendMessage", params=params)
        except:
            token = None

    def _train(self, model: Model, resampler: Resampler, cross_validator: BaseCrossValidator) -> None:
        """
        Trains the provided model using the provided resampler and the instance's data.

        Parameters:
        model (Model): The model to be trained.
        resampler (Resampler): The resampler to be used for resampling the training data.
        """
        name = f"{model}_{resampler}_{cross_validator}"
        multiprocessing.current_process().name = name
        setproctitle.setproctitle(f"Python: {name}")
        try:
            # Choose appropriate features and create synthetic features
            X, y = self._preprocess_data(model, resampler, cross_validator)

            # Perform cross-validation
            iterator = cross_validator.split(X, y)
            if self.cross_validate == False or self.cross_validators is None or self.cross_validators == []:
                iterator = [(np.arange(len(X)), np.arange(len(X)))]

            # Train and test the model
            for fold, (train_i, test_i) in enumerate(iterator):
                for layer in self.layered_models:
                    assert hasattr(layer, "from_storage"), "Layered models must be prefitted."
                    assert layer.from_storage, "Layered models must be prefitted."
                    layer.fit(X[train_i], y[train_i], fold)

                    if self.pass_through_percentage is not None:
                        train_predictions = layer.predict_proba(X[train_i])[:, 1]
                        test_predictions = layer.predict_proba(X[test_i])[:, 1]
                        train_threshold = np.percentile(train_predictions, self.pass_through_percentage * 100)
                        test_threshold = np.percentile(test_predictions, self.pass_through_percentage * 100)
                        train_threshold = train_threshold if train_threshold > 0.5 else 0.5
                        test_threshold = test_threshold if test_threshold > 0.5 else 0.5
                        train_i = train_i[train_predictions > train_threshold]
                        test_i = test_i[test_predictions > test_threshold]
                    else:
                        train_predictions = layer.predict(X[train_i])
                        test_predictions = layer.predict(X[test_i])
                        train_i = train_i[train_predictions == 1]
                        test_i = test_i[test_predictions == 1]
                        
                    print(f"Layered model {layer} reduced train size to {len(train_i)} and test size to {len(test_i)}.")
                
                X_train, y_train = X[train_i, :], y[train_i]
                X_test, y_test = X[test_i, :], y[test_i]
                cv_round = f"cv_{fold}"

                # Add extra resampled data if provided to train the final layer on.
                if self.extra_resampled_train_files is not None:
                    X_extra = pd.read_csv(self.extra_resampled_train_files[fold][0]).to_numpy()
                    y_extra = pd.read_csv(self.extra_resampled_train_files[fold][1]).iloc[:,0].to_numpy()
                    X_train = np.concatenate((X_train, X_extra), axis=0) 
                    y_train = np.concatenate((y_train, y_extra), axis=0)
                    print(f"Added extra resampled data to train set. New size: {len(X_train)}.")

                # Resample the training data
                if not model.skip_resample:
                    X_train, y_train = resampler(X_train, y_train)
                else:
                    self._log(f"Skipping resampling for {model}_{resampler}_{cross_validator}.")

                model.train(X_train, y_train)
                predicted = model.predict(X_test)
                score = model.score(X_test, y_test)

                if self.save_probabilities and hasattr(model.model, "predict_proba"):
                    proba = model.model.predict_proba(X_test)[:, 1]  # type: ignore
                else:
                    proba = np.zeros(len(y_test))
                
                assert len(y_test) == len(predicted), "Predicted and actual Y labels must be the same length."
                assert len(X_test) == len(predicted), "Predicted and actual X labels must be the same length."
                assert len(y_test) == len(X_test), "Both X and Y labels must be the same length."

                self._save_data(model, cross_validator, resampler, model.scorer, fold, (test_i, y_test, predicted, proba))  # type: ignore
                self._log(f"""\nModel: {model}\nCV:{cross_validator}\nCV Round: {cv_round}\nResampler: {resampler}\nOptimization: {model.scorer.__name__}\nScore: {score}\nTag: {self.tag}""")  # type: ignore
                self._save_model(model, cross_validator, resampler, model.scorer, fold)  # type: ignore

        except Exception as e:
            error = f"Error while training {model}_{resampler}_{cross_validator}: {e}"
            raise Exception(error)

    def _preprocess_data(self, model: Model, resampler: Resampler, cross_validator: BaseCrossValidator) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_optimal_features:
            cv_name = BaseStorage.cross_validator_name(cross_validator)
            feature_set = DBStorage(self.records_dir, cv_name, "none", str(model), model.scorer.__name__).load_feature_set()
            # If not all the features are in the dataset, raise an error
            if not set(feature_set).issubset(set(self.X.columns)):
                raise Exception(f"Feature set {feature_set} not in dataset.")
            if len(feature_set) > 0:
                X = self.X[feature_set]
            else:
                X = self.X
        else:
            X = self.X
            print(f"Optimal features not found. Using all features.")

        if self.use_primary_weights:
            X = primary_weight(X, self.y)
            print(f"Using primary weights.")
        if self.use_synthetic_ratios:
            X = create_ratios(X)
            print(f"Using synthetic ratios.")
        if self.use_synthetic_powers:
            X = create_powers(X, columns=X.columns)
            print(f"Using synthetic powers.")

        X = X.to_numpy()
        y = self.y.to_numpy()

        return X, y

    def _log(self, msg: str) -> None:
        """
        Logs the provided data to the output file. TODO - Maybe kill this method or improve it through a rewrite.

        Parameters:
        model (Model): The model used to generate the data (labelling purposes).
        resampler (Resampler): The resampler used to generate the data (labelling purposes).
        data (Tuple): The data to be saved.
        cv_round (int): The cross-validation round number.
        """
        if self.verbosity >= 5:
            with self.log_lock:
                with open(self.log_file, "a") as f:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp}: {msg}")
        if self.verbosity >= 4:
            self.telegram(msg)
        if self.verbosity >= 3:
            print(msg)

    def _save_data(
        self,
        model: Model,
        cross_validator: BaseCrossValidator,
        resampler: Resampler,
        scorer: Callable,
        fold: str,
        data: Tuple,
        tag: Optional[str] = None,
    ) -> None:
        """
        Saves the provided data to a file. The file type is determined by the extension of the output file.

        Parameters:
        model (Model): The model used to generate the data (labelling purposes).
        resampler (Resampler): The resampler used to generate the data (labelling purposes).
        data (Tuple): The data to be saved. The first element is the round label, the rest are the data to be stored.

        Raises:
        AssertionError: If the data is not in the correct format.
        AssertionError: If the output file does not have a valid extension.
        """
        assert type(data) is tuple, "Data must be a tuple."
        assert len(data) > 1, "Data must contain at least 2 elements."
        assert self.output_format in ["csv", "pickle", "both"], "Output type must be valid."

        if tag is None:
            tag = self.tag

        with self.output_lock:
            df = DataFrame(data).transpose()  # type: ignore
            df = df.set_axis(["index", "actual", "predicted", "proba"], axis=1)

            cv_name = BaseStorage.cross_validator_name(cross_validator)
            storage = self.Storage(self.records_dir, cv_name, str(resampler), str(model), scorer.__name__, fold, tag, timestamp=self.timestamp)
            storage.save_result(df)
            storage.save_model(model)

    def _save_model(self, model: Model, cross_validator: BaseCrossValidator, resampler: Resampler, scorer: Callable, fold: str = "0") -> None:
        """
        Saves the provided model to a file.

        Parameters:
        model (Model): The model to be saved.
        resampler (Resampler): The resampler used to generate the model (labelling purposes).
        scorer (str): The scorer used to generate the model (labelling purposes).
        """
        with self.output_lock:
            print(f"Saving model for {model}_{resampler}_{scorer}_{fold}\n")
            cv_name = BaseStorage.cross_validator_name(cross_validator)
            self.Storage(self.records_dir, cv_name, str(resampler), str(model), scorer.__name__, fold, self.tag, timestamp=self.timestamp).save_model(
                model
            )

    def _deserialize_models(self, models: List[Model] | ModelDict, scorers: List[Callable], disable_bayes_search: bool = False):
        """
        Loads the provided models into Model objects. This loads JSON/Dict from the script, not pickle objects.

        Parameters:
        models (List[Model] | ModelDict): The models to be used in the training.
        scorers (Dict[str, str | Callable]): The scorers to be used in the training.
        disable_bayes_search (bool): Whether to disable the use of BayesSearchCV for hyperparameter optimization.
        """
        self.scorers = scorers
        if type(models) == dict:
            self.models = []
            for scoring_method in scorers:
                for name, json in models.items():
                    json["scorer"] = scoring_method
                    assert " " not in name, "Model names cannot contain spaces."
                    assert "/" not in name or "\\" not in name, "Model names cannot contain slashes."
                    self.models.append(Model.from_json(name, json, disable_bayes_search=disable_bayes_search))
        elif type(models) == list:
            self.models = models

    def _deserialize_resamplers(self, resampler_list: ResamplerList = None) -> None:
        """
        Loads the provided resamplers into Resampler objects.
        """
        resamplers: List[Resampler] = []
        if resampler_list is None:
            resampler_list = [None]

        for resampler in resampler_list:
            if resampler is None:
                resamplers.append(Resampler("none"))
            elif resampler.__class__.__name__ == "Resampler":
                resamplers.append(resampler)  # type: ignore
            elif callable(resampler):
                resamplers.append(Resampler(resampler.__name__, resampler))  # type: ignore

        self.resamplers = resamplers

    def _format_result(self, model, resampler, cross_validator, scorer, cv_round, score, tag):
        return f"""Model: {model}
                    CV: {cross_validator}
                    CV Round: {cv_round}
                    Resampler: {resampler}
                    Optimization: {scorer.__name__}
                    Score: {score}
                    Tag: {tag}
                    """
