import pandas as pd
from runtime.analysis import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_results(fnr_cutoff: float = 0, f: Filter = Filter()):
    a = Analysis(records_path=Path("/data/CHML/records"), load_dataframes=False, filter=f)
    a.save_analysis_db()

    data = pd.read_csv(a.analytics_file)
    for tag in data['tag'].unique():
        df = pd.DataFrame(data[data['tag'] == tag])
        means = df[['fnr', 'ppv', 'combination_id']].groupby('combination_id').mean()
        ids = means[means['fnr'] <= fnr_cutoff]
        # Get top ten by ppv
        ids = ids.sort_values('ppv', ascending=False).head(10)
        df = df[df['combination_id'].isin(list(ids.index))]

        if len(df) == 0:
            print(f'No records for {tag}')
            continue
        else:
            print(f'Processing {tag} with {len(df)} records')
        
        df['results'] = df.apply(lambda x: pd.read_csv(x['result_file']), axis=1)

        output = pd.DataFrame()

        results = []

        for index, row in tqdm(df.iterrows(), total=len(df)):
            result_df = row['results']
            row = row.drop('results')
            result_df[row.index] = row
            results.append(result_df)

        results = pd.concat(results)

        yield results, tag

if __name__ == '__main__':
    os.makedirs('/data/CHML/direct_results', exist_ok=True)
#    for i in [33, 452, 449, 286, 365, 425, 434, 439, 368]:
#        f = Filter(tag=f'all_features_layer_1_{i}$')
#        for output, tag in get_results(0.1, f=f):
#            print(f'Saving {tag} with {len(output)} records')
#            output.to_csv(f'./direct_results/{tag}_results.csv')

    f = Filter(tag='^(all_features|basic_ratios|original_data)$')
    f = Filter(tag='^(original_data)$')
    for output, tag in get_results(0.0, f=f):
        print(f'Saving {tag} with {len(output)} records')
        output.to_csv(f'/data/CHML/direct_results/{tag}_results.csv')
