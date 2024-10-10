import os
import warnings

from runtime.resamplers.k_cluster_centroids import KClusterCentroids
warnings.filterwarnings('ignore')

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from runtime.analysis import Analysis, Filter
from runtime.engine import *
from runtime.classifiers.from_storage import ModelFromStorage
from scripts.feature_sets import *

from runtime.resamplers.everything import RESAMPLERS
from runtime.classifiers.everything import MODELS

METRICS: List[Callable] = [
    recall_score,
    f1_score_pos,
    f10_score_pos,
    fp_fn
]

PROTOTYPE_DIR = Path("/data/CHML/prototypes")

def generate_prototypes(combination_id: int, layer: ModelFromStorage, X, y) -> List[Tuple[Path, Path]]:
    """
    Generates the protype points for the given model.
    
    A prototype point is just a synthetic datapoint to be used in training. In this particular case, we are 
    generating prototype points discarded by the first layer (classified as negative) and resampling them to
    be used in the second layer training.
    
    This returns a list of files that contain the prototype points. These files can be passed to the Engine
    to be used in training. They must contain the appropriate columns for the X and y values for a given fold.
    """
    fold_indices = []
    out = []
    
    f = Filter(tag=r"all_features$", cross_validator=r"StratifiedKFold.5$", combination_id=combination_id)
    a = Analysis(records_path=Path("/data/CHML/records"), load_dataframes=False, filter=f)
    a.save_analysis_db()
    analytics = pd.read_csv(a.analytics_file)
    assert len(analytics) == 5  # Working with 5 Folds in this particular script
    for _, row in analytics.iterrows():
        result_df = pd.read_csv(row["result_file"])
        fold_indices.append((row["fold"], result_df.index))

    combination_id = str(combination_id)
    
    # Get indices of the fold points
    for fold, indices in fold_indices:
        print(f"\nID: {combination_id} Fold: {fold} Rows: {indices.shape[0]}")

        os.makedirs(PROTOTYPE_DIR / combination_id, exist_ok=True)
        if Path(PROTOTYPE_DIR / combination_id / f"{fold}_X.csv").exists():
            out.append((PROTOTYPE_DIR / combination_id / f"{fold}_X.csv", 
                        PROTOTYPE_DIR / combination_id / f"{fold}_y.csv"))
            print("Skipping fold, already exists")
            continue

        # Get the samples discarded by the first layer
        layer.fit(X, y, fold)
        predictions = layer.predict(X.loc[indices])
        X_negative = X.loc[indices][predictions == 0]
        Y_negative = y[X_negative.index]
        Y_negative.iloc[0] = 1  # Force the first sample to be positive to avoid resampling issues
        
        # Resample the discarded samples
        print(f"Input Rows: {X_negative.shape[0]}")
        resampler = KClusterCentroids(sampling_strategy=0.001, k=8)   
        X_resampled, y_resampled = resampler.fit_resample(X_negative, Y_negative)
        print(f"Resampled Rows: {X_resampled.shape[0]}")

        # Save the resampled points
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name="definitive_diagnosis")        
        X_resampled = X_resampled[y_resampled != 1]
        y_resampled = y_resampled[y_resampled != 1]
        X_resampled.to_csv(PROTOTYPE_DIR / combination_id / f"{fold}_X.csv", index=False)
        y_resampled.to_csv(PROTOTYPE_DIR / combination_id / f"{fold}_y.csv", index=False)
        
        # Append the files to the output for the engine to load during training
        out.append((PROTOTYPE_DIR / combination_id / f"{fold}_X.csv", 
                    PROTOTYPE_DIR / combination_id / f"{fold}_y.csv"))

    return out
        

for combination_id in [365, 449, 286, 425, 434, 33, 439, 368]: 
    layer_0 = ModelFromStorage(
        Filter(
            combination_id=combination_id,
            tag="all_features$"
        ),
        path=Path("/data/CHML/records")
    )
    
    X, y = load_chml_X_y()
    protoype_files = generate_prototypes(combination_id, layer_0, X, y)
    print(protoype_files)
    
    engine = Engine(MODELS, 
                    RESAMPLERS, 
                    METRICS, 
                    X=X,
                    y=y,
                    tag=f"all_features_prototype_{combination_id}",
                    max_workers=40,
                    verbosity=5,
                    records_dir=Path("/data/CHML/records"),
                    layered_models=[layer_0],
                    extra_resampled_train_files=protoype_files)
    
    engine.run()
