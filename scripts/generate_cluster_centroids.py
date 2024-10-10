import pandas as pd
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import os
from pathlib import Path
import warnings
from runtime.resamplers.k_cluster_centroids import KClusterCentroids
from feature_sets import load_chml_X_y

warnings.filterwarnings('ignore')

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
#

cross_validators = [
    ("StratifiedKFold_5", StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
    ("RepeatedStratifiedKFold_5x2", RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)),
    ("RepeatedStratifiedKFold_5x5", RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42))
]

# Load the dataset
#df = pd.read_csv('../data/training_all_July15.csv', index_col=0)
#y = df['definitive_diagnosis'].to_numpy()                       
#X = df.drop('definitive_diagnosis', axis=1).to_numpy()          

X, y = load_chml_X_y()
X = X.to_numpy()
y = y.to_numpy()

for name, cv in cross_validators:
    for k in [8]: # Repeated K Fold for secondary validation.
#     for k in [11, 5]:
        # Make output directory
        output_dir = Path(f'../resampled_data_May13/cluster_centroids/{k}_means/{name}')
        os.makedirs(output_dir, exist_ok=True)

        iterator = cv.split(X, y)
        for cv_round, (train_i, test_i) in enumerate(iterator):   
            X_train, y_train = X[train_i, :], y[train_i]
            X_test, y_test = X[test_i, :], y[test_i]    
            cv_round = f"cv_{cv_round}"    

            print(f"\nStarting resampling for {name} round {cv_round} with {k} clusters.")

            X_resampled, y_resampled = KClusterCentroids(sampling_strategy=0.01, k=k).fit_resample(X_train, y_train)

            print(f"Original dataset shape {X_train.shape} {y_train.shape}")
            print(f"Resampled dataset shape {X_resampled.shape} {y_resampled.shape}")
            print(f"Finished resampling for {name} round {cv_round}")

            # Save the resampled data
            pd.DataFrame(X_resampled).to_csv(output_dir / f'X_train_{cv_round}.csv')
            pd.DataFrame(y_resampled).to_csv(output_dir / f'y_train_{cv_round}.csv')
            pd.DataFrame(X_test).to_csv(output_dir / f'X_test_{cv_round}.csv')
            pd.DataFrame(y_test).to_csv(output_dir / f'y_test_{cv_round}.csv')
