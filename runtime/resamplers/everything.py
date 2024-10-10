
ratio = 0.02

from skopt.space import  Integer, Categorical, Real
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, ClusterCentroids

from runtime.resampling import Resampler, ResamplerList
from runtime.resamplers.gaussian import GaussianResampler


RESAMPLERS: ResamplerList = [
    None,
    Resampler(name="SMOTE", resampler=SMOTE(sampling_strategy=ratio)),
    Resampler(name="ADASYN", resampler=ADASYN(sampling_strategy=ratio)),
    Resampler(name="BorderlineSMOTE", resampler=BorderlineSMOTE(sampling_strategy=ratio)),
    Resampler(name="RandomOverSampler", resampler=RandomOverSampler(sampling_strategy=ratio)),
    Resampler(name="RandomUnderSampler", resampler=RandomUnderSampler(sampling_strategy=ratio)),
    Resampler(name="NearMiss", resampler=NearMiss(sampling_strategy=ratio)),
    Resampler(name="TomekLinks", resampler=TomekLinks()),
    Resampler(name="Gaussian_FR025_NL001_NPR2", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.01, new_points_ratio=2)), 
#    Resampler(name="Gaussian_FR025_NL001_NPR4", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.01, new_points_ratio=4)), 
#    Resampler(name="Gaussian_FR025_NL050_NPR4", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.50, new_points_ratio=4)), 
#    Resampler(name="Gaussian_FR060_NL001_NPR2", resampler=GaussianResampler(feature_ratio=0.60, noise_level=0.01, new_points_ratio=2)), 
#    Resampler(name="G6aussian_FR060_NL025_NPR4", resampler=GaussianResampler(feature_ratio=0.60, noise_level=0.25, new_points_ratio=4)),
#    Resampler(name="Gaussian_FR060_NL050_NPR1", resampler=GaussianResampler(feature_ratio=0.60, noise_level=0.50, new_points_ratio=1)),  
#    Resampler(name="Gaussian_FR060_NL010_NPR1_100", resampler=GaussianResampler(feature_ratio=0.60, noise_level=0.1, new_points_ratio=multiplier)),   
#    Resampler(name="Gaussian_FR060_NL001_NPR1_100", resampler=GaussianResampler(feature_ratio=0.60, noise_level=0.01, new_points_ratio=multiplier)),  
#    Resampler(name="Gaussian_FR060_NL0001_NPR1_100", resampler=GaussianResampler(feature_ratio=0.60, noise_level=0.001, new_points_ratio=multiplier)),
#    Resampler(name="Gaussian_FR025_NL0001_NPR1_100", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.001, new_points_ratio=multiplier)),
#    Resampler(name="Gaussian_FR025_NL001_NPR1_100", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.01, new_points_ratio=multiplier)),  
#    Resampler(name="Gaussian_FR025_NL010_NPR1_100", resampler=GaussianResampler(feature_ratio=0.25, noise_level=0.1, new_points_ratio=multiplier)),
]
