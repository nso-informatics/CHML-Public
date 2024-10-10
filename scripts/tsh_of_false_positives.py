from runtime.analysis import *
from scripts.feature_sets import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def tsh_of_false_positives(f: Filter, directory: Path):
    # Load the data
    a = Analysis(filter=f, load_dataframes=True, records_path=Path('/data/CHML/records/'))
    combination_id = f.combination_id
    a.save_analysis_db()
    analytics = pd.read_csv(a.analytics_file)
    assert len(analytics) == 5, "Narrow the filter to get only 5 folds"
    
    predictions = pd.DataFrame()
    for i in range(len(analytics)):
        predictions = pd.concat([predictions, pd.read_csv(analytics.iloc[i]['result_file'])], axis=0)
    
    # Rename the columns
    predictions.columns = ['case_index', 'actual', 'predicted', 'probability']

    print("Actual: ", predictions['actual'].value_counts())
    print("Predicted: ", predictions['predicted'].value_counts())

    # set index to the record_id
    fp = ("False_Positive", predictions[(predictions['predicted'] == 1) & (predictions['actual'] == 0)])
    tp = ("True_Positive", predictions[(predictions['actual'] == 1) & (predictions['predicted'] == 1)])
    tn = ("True_Negative", predictions[(predictions['predicted'] == 0) & (predictions['actual'] == 0)])

    X = load_chml()
    X.reset_index(inplace=True)

    for label, scenario in [fp, tp, tn]:
        # Get the indices in the scenario and the data
        data = X[X.index.isin(scenario['case_index'])]

        # Plot the TSH on a histogram
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(data['TSH'], bins=500)
        ax.set_title(f'{label} TSH Distribution -- {combination_id}')
        ax.set_xlabel('TSH')
        ax.set_ylabel('Frequency')
        ax.set_xlim(13, 100)
        ax.set_ylim(0, 20)
        plt.savefig(directory / Path(f'{label}_TSH_distribution.png'))

    # Plot the TSH of the definitive diagnosis
    def label_confusion(predicted, actual):
        if predicted == actual:
            return 'True Positive' if predicted == 1 else 'True Negative'
        else:
            return 'False Positive' if predicted == 1 else 'False Negative'

    predictions['label'] = predictions.apply(lambda row: label_confusion(row['predicted'], row['actual']), axis=1)
    X = X.merge(predictions, left_index=True, right_on='case_index')
    X = X[X['label'] != 'True Negative']

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=X, x='TSH', hue='label', bins=100, multiple='stack')
    ax.set_title(f'TSH Distribution of Predictions -- {combination_id}')
    ax.set_xlabel('TSH')
    ax.set_ylabel('Frequency')
    plt.savefig(directory / Path('predictions_TSH_distribution.png'))
    ax.set_ylim(0, 20)
    plt.savefig(directory / Path('predictions_TSH_distribution_zoom.png'))

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    print(X['definitive_diagnosis'].value_counts())
    ax = sns.histplot(X[X['definitive_diagnosis']==1]['TSH'], bins=500)
    ax.set_title(f'Definitive Diagnosis TSH Distribution -- {combination_id}')
    ax.set_xlabel('TSH')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 50)
    plt.savefig(directory / Path('definitive_diagnosis_TSH_distribution.png'))

    
if __name__ == '__main__':
    base_dir = Path('./TSH_distribution')
    for comb_id in [286, 365, 425, 434, 33, 439, 368, 452, 480, 487, 449, 529]:
        directory = base_dir / str(comb_id)
        directory.mkdir(parents=True, exist_ok=True)
        f = Filter(tag='all_features$', combination_id=comb_id)
        tsh_of_false_positives(f, directory)
    
