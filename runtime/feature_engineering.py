from typing import Dict, List
import sys
import numpy as np
import pandas as pd


def create_ratios(df):
    """
    Create ratios between different columns in the dataframe.
    Only pass in the features you want ratios on.
    Returns the new and old columns.
    """
    
    data = df.copy()
    columns = df.columns
    var_count = len(data.columns)

    if var_count < 2:
        return data
   
    # Separate numeric and non-numeric columns
    # This is done to avoid creating ratios between non-numeric columns
    # Combined at the end to maintain integrity of the data
    non_numeric = data.select_dtypes(exclude=[np.number])
    data = data.select_dtypes(include=[np.number])
    
    def ratio(row):
        replacement_row = pd.Series(np.outer(row, (1 / row).replace([np.inf, -np.inf], 0.0)).flatten())
        row = row.tolist()
        for i in range(var_count):
            replacement_row[i * var_count + i] = row.pop(0)
        return replacement_row
                   
    data = data.apply(ratio, axis=1)
    
    # Rename columns
    for i in range(var_count):
        for j in range(var_count):
            if i == j:
                data.rename(columns={i * var_count + j: f'{columns[i]}'}, inplace=True)
            else:
                data.rename(columns={i * var_count + j: f'{columns[i]}/{columns[j]}'}, inplace=True)
    
    data = pd.concat([data, non_numeric], axis=1)
    
    return data


def create_powers(df: pd.DataFrame, power: int = 2, columns = []) -> pd.DataFrame:
    """
    Create powers of the columns in the dataframe.
    Only pass in the features you want powers on.
    Returns the new and old columns.
    """
    
    data = df.copy()
    var_count = len(columns)
    columns = [col for col in columns if col in data.columns]
    
    assert var_count > 0, 'You need at least 1 variable to create powers'
    assert isinstance(data, pd.DataFrame), 'Data must be a pandas DataFrame'
   
    # Separate numeric and non-numeric columns
    # This is done to avoid creating powers on non-numeric columns
    # Combined at the end to maintain integrity of the data 
    non_numeric = data.select_dtypes(exclude=[np.number])
    data = data.select_dtypes(include=[np.number])
    data[columns] = data[columns].apply(lambda row: row ** power, axis=1)
    data = pd.DataFrame(data)
    
    # Rename columns
    for i in range(var_count):
        data.rename(columns={f"{columns[i]}": f'{columns[i]}'}, inplace=True) 
    
    data = pd.concat([data, non_numeric], axis=1)
        
    return data


def power_set(list_: List[str]) -> List[List[str]]:
    """
    Returns the power set of a list.
    """
    if len(list_) == 0:
        return [[]]
    r = power_set(list_[1:])
    return r + [x + [list_[0]] for x in r]


def primary_weight(X: pd.DataFrame, y: pd.Series, top_n_features: int = 3) -> pd.DataFrame:
    """
    Discard the top n features and create synthetic features from the rest.
    This is based on the algorithm in Web-Based Newborn Screening System for Metabolic Diseases: Machine Learning Versus Clinicians 2013.
    """
    
    
    synthetic = pd.DataFrame()
    
    pos = X[y == 1].median()
    neg = X[y == 0].median()
    original = X.median()
    
    primary_weight_pos = (pos - original) / original
    primary_weight_neg = (neg - original) / original
    
    if len(X.columns) < top_n_features:
        top_n_features = len(X.columns)

    primary_weight_pos = primary_weight_pos.abs().sort_values(ascending=False)[:top_n_features].index
    primary_weight_neg = primary_weight_neg.abs().sort_values(ascending=False)[:top_n_features].index
 
    # Multiply everything together
    for end in [primary_weight_pos, primary_weight_neg]:
        for subset in power_set(end):
            if len(subset) <= 1:
                continue
            synthetic["x".join(subset)] = X[subset].prod(axis=1)
        
    # Take ratios between pos and neg
    for pos_col in primary_weight_pos:
        for neg_col in primary_weight_neg:
            synthetic[f"{pos_col}/{neg_col}"] = X[pos_col] / X[neg_col]

    # Replace inf with max value of a float
    synthetic = synthetic.replace([np.inf, -np.inf], np.finfo(np.float32).max)
    synthetic = synthetic.fillna(0)
    
    return synthetic    
    

if __name__ == "__main__":
    df = pd.DataFrame([
        [1, 1, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], columns=['a', 'b', 'c'])
    
    pd.testing.assert_frame_equal(create_ratios(df), pd.DataFrame([
        [1.0, 1.0, 1/3, 1.0, 1.0, 1/3, 3.0, 3.0, 3.0],
        [4.0, 4/5, 4/6, 5/4, 5.0, 5/6, 6/4, 6/5, 6.0],
        [7.0, 7/8, 7/9, 8/7, 8.0, 8/9, 9/7, 9/8, 9.0]
    ], columns=['a', 'a/b', 'a/c', 'b/a', 'b', 'b/c', 'c/a', 'c/b', 'c']))

    pd.testing.assert_frame_equal(create_powers(df, 2, ['a', 'b', 'c']), pd.DataFrame([
        [1, 1, 9],
        [16, 25, 36],
        [49, 64, 81]
    ], columns=['a', 'b', 'c']))
    
    pd.testing.assert_frame_equal(create_powers(df, 3, ['a', 'b', 'c']), pd.DataFrame([
        [1, 1, 27],
        [64, 125, 216],
        [343, 512, 729]
    ], columns=['a', 'b', 'c']))

    
