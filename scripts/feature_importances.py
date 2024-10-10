from runtime.analysis import *
from runtime import *
from pickle import load
from scripts.feature_sets import *
from time import sleep

def feature_importances(f: Filter, feature_labels):
    sleep(1) # Sleep to prevent duplication of analysis file names
    a = Analysis(filter=f, records_path=Path('/data/CHML/records/'))
    results = a.save_analysis_db()
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    models = []
    results['model_file'].apply(lambda x: models.append(load(open(x, 'rb'))))
    feature_importances = pd.DataFrame()
    importance = []
    try:
        for model in models:
            importance.append(pd.DataFrame(model.model.best_estimator_.feature_importances_, index=feature_labels))
    except AttributeError:
        return None

    feature_importances[f.combination_id] = pd.concat(importance).groupby(level=0).mean()

    return feature_importances

if __name__ == '__main__':
    # Load the model
    importances = []
    feature_labels = load_chml_X_y()[0].columns 
    for i in [286, 365, 425, 434, 33, 439, 368, 452, 480, 487, 449, 529]:
        f = Filter(tag='all_features$', combination_id=i)
        result = feature_importances(f, feature_labels)
        if result is not None:
            importances.append(result)

    importances = pd.concat(importances, axis=1)

    print(importances)
    importances.to_csv('feature_importances.csv')

