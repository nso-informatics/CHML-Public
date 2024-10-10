import sys
import os
import pandas as pd
import numpy as np

sys.path.append('..')


from scripts.feature_sets import load_chml_X_y

if __name__ == '__main__':
    X, y = load_chml_X_y()
    df = pd.concat([X, y], axis=1)
    
    # Calculate the 5 number summary of the dataset
    summary = df.describe()
    
    # Save summary to a csv file
    summary.to_csv('five_number_summary.csv')
    
    
