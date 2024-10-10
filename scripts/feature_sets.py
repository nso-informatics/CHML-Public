from typing import Optional, Tuple
import pandas as pd
from pathlib import Path
from typing import List, Optional

from runtime.analysis import Analysis, Filter

DATA_PATH = Path("/home/adefuria/adefuria/CH-ML/preprocessing/processed.csv")
DEFINITIVE_OUTCOMES = Path("../preprocessing/combined.csv")

def get_episode_from_index(index: int | List[int], df: Optional[pd.DataFrame]) -> pd.Series:
    df = load_chml() if df is None else df
    df = df.reset_index()
    return df.loc[index]

def get_definitive_diagnosis_from_episode(episodes: str | List[str]) -> pd.DataFrame:
    df = pd.read_csv(DEFINITIVE_OUTCOMES)
    # TODO confirm the column name
    return df[df['episode'].isin(episodes)][['definitive_diagnosis', 'episode']]

def load_chml(include_episode: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    assert type(df) == pd.DataFrame
    assert len(df) > 500000, "Confirm you are using the whole dataset" # TODO remove
    
    # Drop episode that should not be in dataset
    indices = (df['TSH'] < 14) & (df['definitive_diagnosis'])
    print(f"Dropping {indices.sum()} rows with TSH < 14 and Definitive Diagnosis")
    print(f"{df['episode'][indices].values}")
    df = df[~indices]
    df = df if include_episode else df.drop(columns=["episode"])
    
    return df

def load_chml_X_y() -> Tuple[pd.DataFrame, pd.Series]:
    df = load_chml()
    y = df["definitive_diagnosis"]
    X = df.drop(columns=["definitive_diagnosis"])
    assert type(X) == pd.DataFrame
    assert type(y) == pd.Series
    return X, y

