import os
import sys

sys.path.append('..')
sys.path.append('../../')

from runtime.engine import *
from runtime.resamplers.gaussian import *
from runtime.evaluation import *
from runtime.resamplers import *
from scripts.feature_sets import *
import warnings
warnings.filterwarnings('ignore')

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"


from runtime.resamplers.everything import RESAMPLERS
from runtime.classifiers.everything import MODELS

RESAMPLERS = RESAMPLERS + [
    Resampler(name="Gaussian_FR025_NL005_NPR2", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.05, new_points_ratio=2)),
    Resampler(name="Gaussian_FR075_NL005_NPR2", resampler=GaussianResampler(feature_ratio=0.75, noise_level=0.05, new_points_ratio=2)),
    Resampler(name="Gaussian_FR025_NL005_NPR5", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.05, new_points_ratio=5)),
]

METRICS = [
    f1_score_pos,
    fp_fn,
    recall_score,
    f10_score_pos,
]

cross_validators = [
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
]

X, y = load_chml_X_y()
X = X.drop(columns=["tsh_squared", "tsh_cubed", "weight_to_gestational_age", "weight_to_age_at_collection"])

engine = Engine(MODELS,                                               
                RESAMPLERS,                                           
                METRICS,                                              
                X=X,                                     
                y=y,                                   
                cross_validators=cross_validators,                    
                max_workers=12,
                tag="original_data",  
                records_dir=Path("/data/CHML/records/"),
                verbosity=5)                                          
engine.run()                                                          

