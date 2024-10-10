import pandas as pd
import json
import os
from runtime.analysis import Analysis, Filter
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

f = Filter(
#    cross_validator="Repeated(.*)",
#    score="f1(.*)",
#    model="Balanced Random Forest(.*)",
#    tag=r"layer"
#    tag=r"baseline(.*)"
)

#a = Analysis(filter=f, load_models=True)
a = Analysis(records_path=Path("/data/CHML/records"), load_dataframes=False, filter=f)
#a.load_data()
a.save_analysis_db()
print(a.analytics_file)



if len(sys.argv) >= 2:
    if sys.argv[1] == "scp":
        os.system(f"scp {a.analytics_file} alexander@alexanderdefuria.ddns.net:/home/alexander/")
    elif sys.argv[1] == "print":
        print(pd.read_csv(a.analytics_file))
