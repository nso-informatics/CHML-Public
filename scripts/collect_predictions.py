from runtime.analysis import *

def most_commonly_misclassified(f: Filter):
    a = Analysis(records_path=Path("/data/CHML/records"), load_dataframes=True, filter=f)


if __name__ == '__main__':
    output = pd.DataFrame()
    for i in [33, 452, 449, 286, 365, 425, 434, 439, 368]:
        f = Filter(tag=f'all_features_layer_1_{i}$')
        most_commonly_misclassified(f)
        
    
    output.to_csv(f'./direct_results/{tag}_results.csv')