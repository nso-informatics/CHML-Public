from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer
from skopt.space import Real, Integer, Categorical
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
from engine import *

# df = pd.read_csv("/home/chml/whowe/training_all_July15.csv")
# X = df.drop("Definitive Diagnosis", axis=1).drop("Episode", axis=1).values
# y = df["Definitive Diagnosis"].values
X, y = load_breast_cancer(return_X_y=True)
assert type(X) == np.ndarray
assert type(y) == np.ndarray

MODELS = {
    "BalancedBagging": {
        "estimator": BalancedBaggingClassifier(),
        "search_spaces": {
            "warm_start": Categorical([True, False]),
            "bootstrap": Categorical([True, False]),
            "n_estimators": Integer(5, 50),
#           "sampling_strategy": Real(0.01, 1, prior="log-uniform"), # Controls sampling rate for RUS.
        },
    },
}

RESAMPLERS = [
    None,
    SMOTE(),
]

METRICS: List[Callable] = [
    f1_score_pos,
    f1_score_neg,
    accuracy,
]

engine = Engine(MODELS, 
                RESAMPLERS, 
                METRICS, 
                X=X,
                y=y,
                tag="breast_cancer",
                max_workers=40,
                verbosity=5)

print(load_breast_cancer())
#engine.run()
