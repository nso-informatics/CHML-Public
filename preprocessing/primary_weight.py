import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('processed.csv')
    
    df.drop("episode", axis=1, inplace=True)
    
    pos = df[df['definitive_diagnosis'] == 1].median()
    neg = df[df['definitive_diagnosis'] == 0].median()
    original = df.median()
    
    primary_weight_pos = (pos - original) / original
    primary_weight_neg = (neg - original) / original
    
    # Get the top 10 features
    primary_weight_pos = primary_weight_pos.abs().sort_values(ascending=False)
    primary_weight_neg = primary_weight_neg.abs().sort_values(ascending=False)
    
    primary_weight_pos.to_csv('primary_weight_pos.csv')
    primary_weight_neg.to_csv('primary_weight_neg.csv')
    
    
    
    
    
    
    