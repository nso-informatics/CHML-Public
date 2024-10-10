import pandas as pd
import numpy as np
import sweetviz as sv
from runtime.analysis import *
from tqdm import tqdm, tqdm_pandas
import warnings
import os
from pathlib import Path
from scripts.feature_sets import *
warnings.filterwarnings("ignore")

tqdm.pandas()

def get_false_negatives(f: Filter = Filter()):
    a = Analysis(records_path=Path("/data/CHML/records"), load_dataframes=False, filter=f)
    a.save_analysis_db()

    data = pd.read_csv(a.analytics_file)
    for tag in data['tag'].unique():
        df = pd.DataFrame(data[data['tag'] == tag])

        if len(df) == 0:
            print(f'No records for {tag}')
            continue
        else:
            print(f'Processing {tag} with {len(df)} records from {tag}')
        
        df = df[['result_file', 'combination_id', 'tag']]
        df['results'] = df.progress_apply(lambda x: pd.read_csv(x['result_file']), axis=1)
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            result_df = row['results']
            row = row.drop('results')
            result_df[row.index] = row
            results.append(result_df[['index', 'actual', 'predicted', 'combination_id', 'tag']])

        output = pd.concat(results)

        for combination_id, combination in output.groupby('combination_id'):
            # Get false negatives
            fn = combination[(combination['actual'] == 1) & (combination['predicted'] == 0)]
            if len(fn) not in range(1, 7): continue # False negative rate under 2% is acceptable for this analysis
            fn.rename(columns={'index': 'case_index'}, inplace=True) # type: ignore
            fn['case_index'] = fn['case_index'].astype(int)
            fn = fn[['case_index', 'combination_id', 'tag']]
            fn.reset_index(drop=True, inplace=True) # type: ignore
#            print(f'Found {len(fn)} false negatives for {tag} with combination_id {combination_id}')
            yield fn, tag, f.combination_id
        


if __name__ == '__main__':
 
    df = load_chml(include_episode=True)
    df.reset_index(drop=True, inplace=True)

    report = sv.analyze(df[df['TSH'] > 100], pairwise_analysis='auto', target_feat='definitive_diagnosis')
    report.show_html('/data/CHML/problem_points/high_tsh_report.html', open_browser=False)

    results = []
    for fn, tag, combination_id in get_false_negatives(f=Filter(tag='all_features')): results.append(fn)
    fn_df = pd.concat(results)
    print(fn_df) 
    os.makedirs('/data/CHML/problem_points', exist_ok=True)
    fn_df.to_csv('/data/CHML/problem_points/false_negatives.csv', index=False)

    fn_summary = fn_df.groupby('case_index').count()
    fn_summary = fn_summary[['combination_id']]
    fn_summary['case_index'] = fn_summary.index
    fn_summary.rename(columns={'combination_id': 'count'}, inplace=True)
    fn_summary.to_csv('/data/CHML/problem_points/false_negatives_summary.csv') # We use this to ensure that we have results regardless of the sort and merge 
    original_len = len(fn_summary)
    fn_summary = fn_summary.sort_values('count', ascending=False)
    fn_summary = pd.merge(fn_summary, df, left_index=True, right_index=True, how='inner')
    assert len(fn_summary) == original_len
    fn_summary[['count', 'episode', 'TSH', 'definitive_diagnosis']].to_csv('/data/CHML/problem_points/false_negatives_summary.csv')
    print(fn_summary[['case_index', 'count', 'definitive_diagnosis']])

    tp_minus_fn = df[~df.index.isin(fn_summary['case_index'])]
    report = sv.compare(source=[tp_minus_fn, "TP - FN"], compare=[df, "FN"], target_feat='definitive_diagnosis' )
    report.show_html('/data/CHML/problem_points/false_negatives_report.html', open_browser=False)
