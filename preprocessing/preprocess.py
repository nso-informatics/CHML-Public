import os
from pathlib import Path
import re
from typing import Callable, Dict
import numpy as np
import pandas as pd
from datascroller import scroll
import sweetviz as sv
import matplotlib.pyplot as plt
from tqdm import tqdm
from runtime.feature_engineering import create_ratios, create_powers

ORIGIN = Path("combined.csv")
DATA_DIR = Path("/home/chml/adefuria/data/initials_and_outcomes")

metabolic_columns = [
    "ALA", "ARG", "CIT", "GLY", "LEU", "MET", "ORN", "PHE", "SUAC", "TYR", "VAL",
    "C0", "C2", "C3", "C3DC", "C4", "C4DC", "C4OH", "C5", "C5:1", "C5OH", "C5DC",
    "C6", "C6DC", "C8", "C8:1", "C10", "C10:1", "C12", "C12:1", "C14", "C14:1",
    "C14:2", "C14OH", "C16", "C16OH", "C16:1OH", "C18", "C18:1", "C18:2", "C18OH",
    "C18:1OH", "BIOT", "GALT", "IRT", "TREC_QN", "TSH", "A", "F", "F1", "FAST",
]

"""
This code is largely self-documenting but there is an accompanying 
`preprocessing.md` file that explains the "why" behind the "what" in this script.
"""
def preprocess() -> pd.DataFrame:
    data = load_omni()
#    data = pd.read_csv(ORIGIN)
    data = map_names(data)
    data = clean_values(data)
    data = handle_missing_values(data)
    data = make_calculations(data)
    data = one_hot_encode(data)
    data = integer_encode(data)
    
    data.to_csv("./new_processed.csv", index=False)

    # generate_statistics(data)
    # return data

    
def generate_statistics(data: pd.DataFrame) -> None:
    """
    Generate statistics and reports based on the given dataset using Sweetviz.
    Save the reports to the current directory.
    """

    # Split into two datasets based on definitive_diagnosis
    positive = data[data["definitive_diagnosis"] == 1]
    negative = data[data["definitive_diagnosis"] == 0]

    # Generate reports
    report = sv.compare([positive, "Positive"], [negative, "Negative"]) # type: ignore
    report.show_html("./compare_dataset.html", open_browser=False)

    report = sv.analyze(data, target_feat="definitive_diagnosis", pairwise_analysis="on")
    report.show_html("./dataset.html", open_browser=False, )   
    
    #scroll(data) # Interactive data exploration


def load_omni() -> pd.DataFrame:
    """
    Our source of truth is a combination of two sheets in the OMNI dataset, an excel file.
    We combine the outcomes and screening_data sheets into a single dataframe.
The outcomes sheet contains the definitive diagnosis for each screen positive episode.

def preprocess():
    The screening_data sheet contains the screening data for each episode.
    """
    if not os.path.exists(ORIGIN):
        concat_excel_files()
        omni = pd.read_csv(ORIGIN)
        omni['Definitive Diagnosis'] = omni['Definitive Diagnosis'].fillna("negative").astype(str)
        omni.to_csv(ORIGIN, index=False)
    else:
        omni = pd.read_csv(ORIGIN, low_memory=False, )
 
    return omni


def map_names(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns in the dataset to match the target schema.
    Any columns not in the target schema are removed.
    """
    column_names = {
        # Original: New
        "Episode": "episode",
        "Definitive Diagnosis": "definitive_diagnosis",
        "Gestational Age (days)": "gestational_age",
        "Birth Weight (g)": "birth_weight",
        "Sex": "sex",
        "Age at Collection (hr)": "age_at_collection",
        "Transfusion Status": "transfusion_status",
        "Multiple Birth Rank": "multiple_birth_rank",
        "ALA_I (RAW)": "ALA",
        "ARG_I (RAW)": "ARG",
        "CIT_I (RAW)": "CIT",
        "GLY_I (RAW)": "GLY",
        "LEU_I (RAW)": "LEU",
        "MET_I (RAW)": "MET",
        "ORN_I (RAW)": "ORN",
        "PHE_I (RAW)": "PHE",
        "SUAC_I (RAW)": "SUAC",
        "TYR_I (RAW)": "TYR",
        "VAL_I (RAW)": "VAL",
        "C0_I (RAW)": "C0",
        "C2_I (RAW)": "C2",
        "C3_I (RAW)": "C3",
        "C3DC_I (RAW)": "C3DC",
        "C4_I (RAW)": "C4",
        "C4DC_I (RAW)": "C4DC",
        "C4OH_I (RAW)": "C4OH",
        "C5_I (RAW)": "C5",
        "C5:1_I (RAW)": "C5:1",
        "C5OH_I (RAW)": "C5OH",
        "C5DC_I (RAW)": "C5DC",
        "C6_I (RAW)": "C6",
        "C6DC_I (RAW)": "C6DC",
        "C8_I (RAW)": "C8",
        "C8:1_I (RAW)": "C8:1",
        "C10_I (RAW)": "C10",
        "C10:1_I (RAW)": "C10:1",
        "C12_I (RAW)": "C12",
        "C12:1_I (RAW)": "C12:1",
        "C14_I (RAW)": "C14",
        "C14:1_I (RAW)": "C14:1",
        "C14:2_I (RAW)": "C14:2",
        "C14OH_I (RAW)": "C14OH",
        "C16_I (RAW)": "C16",
        "C16OH_I (RAW)": "C16OH",
        "C16:1OH_I (RAW)": "C16:1OH",
        "C18_I (RAW)": "C18",
        "C18:1_I (RAW)": "C18:1",
        "C18:2_I (RAW)": "C18:2",
        "C18OH_I (RAW)": "C18OH",
        "C18:1OH_I (RAW)": "C18:1OH",
        "BIOT_I (RAW)": "BIOT",
        "GALT_I (RAW)": "GALT",
        "IRT_I (RAW)": "IRT",
        "TREC QN_I (RAW/CALC)": "TREC_QN",
        "TSH_I (RAW)": "TSH",
        "HGB Pattern_I (RAW)": "HGB_Pattern",
        "A_I (RAW)": "A",
        "F_I (RAW)": "F",
        "F1_I (RAW)": "F1",
        "FAST_I (RAW)": "FAST",
    }

    # Remove columns that are not in the column_names dictionary, rename listed columns.
    removed_columns = [x for x in dataset.columns if x not in column_names.keys()]
    dataset.drop(columns=removed_columns, inplace=True)
    dataset.rename(columns=column_names, inplace=True)
    return dataset


def handle_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values
    
    Perform analysis of which rows are missing values and which columns have missing values.
    """
    
    dataset = dataset.dropna()
    return dataset

def make_calculations(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate new columns based on existing columns. 
    
    For example, weight_to_age is a ratio calculated by dividing birth_weight by age.
    or F1+FAST is a sum of F1 and FAST.
    
    The new columns are added to the dataset.
    """
    new_columns = {
        # Label: Calculation
        'weight_to_age_at_collection': lambda x: x['birth_weight'] / x['age_at_collection'],
        'weight_to_gestational_age': lambda x: x['birth_weight'] / x['gestational_age'],
        'tsh_squared': lambda x: x['TSH'] ** 2,
        'tsh_cubed': lambda x: x['TSH'] ** 3,
    }
    
    for column, calculation in tqdm(new_columns.items(), desc="Calculating new columns"):
        dataset[column] = calculation(dataset)
    
    return dataset


def clean_values(dataset: pd.DataFrame) -> pd.DataFrame :
    """
    Clean values in the dataset.
    
    For example, we can convert strings to integers, floats, or booleans.
    We can also drop rows based on the values in the dataset.
    Any row with NaN values will be dropped.
    """
    
    numeric = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$") # Regular expression for numeric values (integers, floats, scientific notation)
    
    # Rules to handle all values in the dataset
    # Default ensure that all values are numeric, drop rows with missing values
    default_behaviour: Dict[re.Pattern, Callable] = {
        numeric: lambda x: abs(float(x)),
        re.compile(r"^(Not Tested)$"): lambda _: None,
    }
    
    rules: Dict[str, Dict[re.Pattern, Callable]] = {
        "episode": {
            re.compile(r".*"): lambda x: x,
        },
        "definitive_diagnosis": {
            re.compile(r"(CH)"): lambda _: 1,
            re.compile(r".*negative.*"): lambda _: 0,
            re.compile(r"(Not Affected)$"): lambda _: 0,
            re.compile(r"(Transient)$"): lambda _: 0,  
            re.compile(r"(Deceased)$"): lambda _: 0,
            re.compile(r"(DERF Pending)$"): lambda _: None, # Drop rows with DERF Pending. These could be positives or negatives.
            re.compile(r"(DERF PENDING)$"): lambda _: None, # Drop rows with DERF PENDING. These could be positives or negatives.
            re.compile(r"(Maternal Disease: Maternal PTU/Graves Disease)$"): lambda _: 0,
            re.compile(r"(hypothyroidism secondary to amiodarone)$"): lambda _: 0,
            re.compile(r"(Iodine exposure)$"): lambda _: 0,
            re.compile(r"(Deceased \(please specify cause of death\))$"): lambda _: 0,
        },
        "gestational_age": {
            numeric: lambda x: abs(int(x)),
        },
        "birth_weight": {
            numeric: lambda x: abs(int(x)),
        },           
        "sex": {
            re.compile(r".*(M|F).*"): lambda x: 0 if 'fe' in str(x).lower() else 1,
            re.compile(r".*(Ambiguous).*"): lambda _: 2, # Ambiguous is mapped into M=1 and F=1
            re.compile(r".*(Unknown).*"): lambda _: 3, # Unknown is mapped into M=0 and F=0
        },
        "age_at_collection": {
            numeric: lambda x: abs(float(x)) if abs(float(x)) > 24.0 and abs(float(x)) < 168.0 else None, # Drop rows with age outside of 1-7 days 
        },
        "transfusion_status": {
            re.compile(r"N"): lambda _: 0,
            re.compile(r"Y"): lambda _: 1,
            re.compile(r"U"): lambda _: 0,
        },
        "multiple_birth_rank": {
            re.compile(r"^(nan)$"): lambda _: 0,
            # There are issues with categorical values and sweetvix. 
            # Best to leave as boolean for now. Also do not know the impact
            # of multiple births on the outcome. (bool vs cat.)
#            re.compile(r"^[a-zA-Z]$"): lambda x: ord(str(x).lower()) - 97 if str(x).isalpha() else None
            re.compile(r"^[a-zA-Z]$"): lambda _: 1
        },
        "HGB_Pattern": {
            re.compile(r"^FA$"): lambda _: 1,
            re.compile(r".*"): lambda _: 0,
        },
    }
        
    # Set all columns to default rules and update with specific rules
    mapping: Dict[str, Dict] = {}
    for column in dataset.columns:
        mapping[column] = default_behaviour
    mapping.update(rules)
   
    # Apply the first rule that matches the value in the column
    def apply_rule(x, rule):
        for key, value in rule.items():
            if key.match(str(x)):
                return value(x)
        # If no rule matches, we drop the row
        return None
    
    for column in tqdm(mapping, desc="Cleaning values"):
        dataset[column] = dataset[column].apply(lambda x: apply_rule(x, mapping[column]))

    # These are anomalies in the dataset that need to be removed
    # Positive cases with a TSH < 14 should not be in the dataset as they were screen negative
    dataset = dataset[~(dataset['TSH'] < 14) & (dataset['definitive_diagnosis'])]

    dataset.to_csv("./cleaned_with_nulls.csv", index=False)
    
    return dataset


def one_hot_encode(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one hot encoding on the dataset by mapping values into boolean columns.
    """
    columns: Dict[str, Dict] = {
        # This is an example of how to one hot encode a column
        # "column": {
        #    value: label,
        #    value: label
        #}
        "sex": {
            0: ["sex_male"],
            1: ["sex_female"],
            2: ["sex_male", "sex_female"],
        }
    }
    
    # One hot encode columns using the columns dictionary
    new_columns = []
    for column, mapping in columns.items():
        for value, labels in mapping.items():
            for label in labels:
                new_columns.append(label)

    dataset[new_columns] = np.zeros((len(dataset), len(new_columns)))

    for column, mapping in columns.items():
        for row in tqdm(dataset.iterrows(), desc=f"One hot encoding {column}", total=len(dataset)):
            for value, labels in mapping.items():
                for label in labels:
                    if row[1][column] == value:
                        dataset.at[row[0], label] = 1                

        dataset.drop(columns=[column], inplace=True)
    return dataset


def integer_encode(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float columns to integer columns if all values can be represented as integers.
    """
    for column in dataset.columns:
        if dataset[column].dtype == float:
            if np.all(dataset[column] == dataset[column].astype(int)):
                dataset[column] = dataset[column].astype(int)
            
    return dataset


def concat_excel_files():
    """
    Concatenate all excel files in the directory into a single file.
    """
    screening_files = [file.name for file in DATA_DIR.glob(r"*") if re.match(r'.*\d{8,8}-\d{8,8}_OMNINBSInitialRAWAnalytesAndScrDets_.*\.xlsx', str(file))]
    outcome_files = [file.name for file in DATA_DIR.glob(r"*") if re.match(r'.*\d{8,8}-\d{8,8}_CHDiagnosticOutcomes_.*\.xlsx', str(file))]
    screening_data = pd.DataFrame()
    outcomes = pd.DataFrame()
    for file in screening_files:
        file = DATA_DIR / file
        screening = pd.read_excel(file)
        screening.rename(columns={"Accession Number": "Episode"}, inplace=True)
        screening_data = pd.concat([screening_data, screening], ignore_index=True)

    for file in outcome_files:
        file = DATA_DIR / file
        outcomes_data = pd.read_excel(file)
        outcomes_data.rename(columns={"Accession Number": "Episode"}, inplace=True)
        outcomes = pd.concat([outcomes, outcomes_data], ignore_index=True)
    
    print("Screening Data Shape: ", screening_data.shape)
    print("Outcomes Shape: ", outcomes.shape)

    data = screening_data.merge(outcomes, on="Episode", how="left")
    data.to_csv(ORIGIN, index=False)

if __name__ == "__main__":
    preprocess()
