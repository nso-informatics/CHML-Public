"""
pca.py

This module performs Principal Component Analysis (PCA) on a dataset.

PCA is a dimensionality reduction technique that is commonly used in machine learning and data visualization.
It uses the concept of eigenvectors and eigenvalues to transform the data to a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

Begin with preprocess.py and analyze.py to understand the flow of the program and the nature of the dataset.
"""

from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

ORIGIN = Path('processed.csv')

def pca(df: pd.DataFrame):
    """
    Perform PCA on a dataset.

    Returns:
    - The principal components of the dataset.
    """
    df.drop("episode", axis=1, inplace=True)

    pos = df[df['definitive_diagnosis'] == 1]
    neg = df[df['definitive_diagnosis'] == 0]
    
    pos_pca = PCA(n_components=2).fit(pos)
    neg_pca = PCA(n_components=2).fit(neg)
    
    for component in range(pos_pca.n_components_): 
        pc = pd.DataFrame(pos_pca.components_[component], index=pos.columns, columns=['Positive'])
        pc = pd.concat([pc, pd.DataFrame(neg_pca.components_[component], index=neg.columns, columns=['Negative'])])
        
        # Plot the principal components
        plt.figure(figsize=(20, 10))
        plt.bar(pc.index, pc['Positive'], color='b', label='Positive')
        plt.bar(pc.index, pc['Negative'], color='r', label='Negative')
        plt.xlabel('Index')
        plt.ylabel('Principal Component')
        plt.title('Principal Components of Dataset')
        plt.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"pca_{component}.png")
    
    

if __name__ == '__main__':
    df = pd.read_csv(ORIGIN)
    pca(df)



