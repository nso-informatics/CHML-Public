
"""
Evaluation metrics for binary classification

These are functions that take a model, X, and y as input and return a score
Functions are defined here so that they can be used in the `METRICS` dictionary.
They must be defined as full functions and not lambdas because sklearn doesn't know what to do with lambdas (best guess).
They are redfined over the sklearn methods because the form required by BayesSearchCV is different.

General Form:
def metric(model: BaseEstimator, X: ArrayLike, y: ArrayLike) -> float:
    return score
"""

# Import only the metrics to not spill over into script and allow duplicate naming.
from typing import Any, Callable, Union
from numpy import ndarray
from sklearn import metrics
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator

type Metric = Callable[[BaseEstimator, ArrayLike, ArrayLike], float | ndarray[Any, Any]]

def fp_fn(model, X, y, fp_weight: float = 1.0, fn_weight: float = 1.0) -> float:
    """
    Weighted sum of the inverse of the false positive and false negative rates.
    The weights are used to balance the importance of the two rates.
    `fp_rate *= fp_weight` and `fn_rate *= fn_weight`
    """
    # Get the confusion matrix
    cm = metrics.confusion_matrix(y, model.predict(X))
    # Get the false positive and false negative rates
    false_positive_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    # Clip the rates to prevent division by zero
    epsilon = 1e-8
    false_positive_rate = max(false_positive_rate, epsilon)
    false_negative_rate = max(false_negative_rate, epsilon)

    # Weight the rates
    false_positive_rate *= fp_weight
    false_negative_rate *= fn_weight
    
    return 1 / (false_positive_rate + false_negative_rate)

def recall_score(model, X, y) -> float:
    """
    True positive rate (recall)
    """
    return float(metrics.recall_score(y, model.predict(X)))

def f1_score_pos(model, X, y):
    return metrics.f1_score(y, model.predict(X), pos_label=1)

def f2_score_pos(model, X, y):
    return metrics.fbeta_score(y, model.predict(X), beta=2, pos_label=1)

def f4_score_pos(model, X, y):
    return metrics.fbeta_score(y, model.predict(X), beta=4, pos_label=1)

def f10_score_pos(model, X, y):
    return metrics.fbeta_score(y, model.predict(X), beta=10, pos_label=1)

def f1_score_neg(model, X, y):
    return metrics.f1_score(y, model.predict(X), pos_label=0)

def f1_score_macro(model, X, y):
    return metrics.f1_score(y, model.predict(X), average="macro")

def accuracy(model, X, y):
    return metrics.accuracy_score(y, model.predict(X))
