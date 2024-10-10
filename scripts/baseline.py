import os

import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from imblearn.ensemble import BalancedBaggingClassifier

from runtime import *
from runtime.analysis import Analysis, Filter
from runtime.engine import Engine
from runtime.evaluation import *
from runtime.resampling import *
from runtime.classifiers.traditional_ch_screen import TraditionalScreen

from feature_sets import *

RESAMPLERS: ResamplerList = [
    None,
]

METRICS =[ 
    f1_score_pos,
]

data = load_chml()
y = data['definitive_diagnosis']
X = pd.DataFrame(data['TSH'])

def generate_numbers():
    number = 10.0
    while number <= 25:
        number = float("{:.3f}".format(round(number, 3)))
        yield number
        number += 0.1

for i in generate_numbers():
    MODELS = {
        'baseline': {
            'estimator': TraditionalScreen(tsh_index = 0, cutoff = i),
            'search_spaces': {}
        }
    }
    
    engine = Engine(MODELS,
                    RESAMPLERS,
                    METRICS,
                    records_dir=Path("/data/CHML/records/"),
                    X=X,
                    y=y,
                    tag=f"baseline_ge_{i}",
                    use_database=True,
                    use_optimal_features=False,
                    disable_bayes_search=True,
                    verbosity=5)
    engine.run()


#f = Filter(tag="baseline")
#analysis = Analysis()
#analysis.load_data()
#print(analysis.analytics_file)
