import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os
import warnings

warnings.filterwarnings("ignore")


def calculate_ppv(values: np.ndarray, actual: np.ndarray) -> float:
    values = values.astype(bool)
    actual = actual.astype(bool)
    tp = np.sum(values & actual)
    fp = np.sum(values & ~actual)
    return float(tp / (tp + fp))


def and_combination(df: pd.DataFrame) -> pd.DataFrame:
    results = [(combination_id, df[["case_index", "actual", "predicted", "proba"]]) for combination_id, df in df.groupby("combination_id")]
    results = [(combination_id, df.sort_values("case_index").reset_index()) for combination_id, df in results if df.shape == (616910, 4)]
    results = dict(results)
    combinations = []
    print(f"Found {len(results)} combinations")
    for i in range(1, len(results) + 1):
        combinations += itertools.combinations(results.keys(), i)

    print(f"Found {len(combinations)} combinations")

    combined_results = []

    for combination in combinations:
        current = results[combination[0]].copy()
        current["predicted"] = current["predicted"].astype(bool)
        for combination_id in combination:
            current["predicted"] = current["predicted"] & results[combination_id]["predicted"].astype(bool)

        # Evaluate the results PPV
        tp = current[(current["predicted"] == True) & (current["actual"] == True)].shape[0]
        fp = current[(current["predicted"] == True) & (current["actual"] == False)].shape[0]
        ppv = tp / (tp + fp)

        combined_results.append((combination, len(combination), ppv, 0.0))

    combined_results = pd.DataFrame(combined_results, columns=["combination", "model_count", "ppv", "fnr"])

    combined_results.sort_values(["ppv", "model_count"], ascending=[False, True])
    # Plot the results

    return combined_results


def simple_vote(df: pd.DataFrame) -> pd.DataFrame:
    results = [
        (combination_id, df[["case_index", "actual", "predicted", "proba"]].set_index("case_index"))
        for combination_id, df in df.groupby("combination_id")
    ]
    results = dict(results)

    for combination_id, cdf in results.items():
    #     cdf["predicted"] = cdf["predicted"].astype(bool)
    #     cdf["actual"] = cdf["actual"].astype(bool)
    #     cdf["proba"] = cdf["proba"].astype(float)

    #     cdf = cdf[cdf["predicted"] == True]
    #     cdf["proba"] = (cdf["proba"] - cdf["proba"].mean()) / cdf["proba"].std()  # 20.1 PPV
    #     # cdf["proba"] = cdf["proba"] / len(cdf["proba"])  
    #     # cdf["proba"] = cdf["proba"] - cdf["proba"].min()  # 20.7 PPV
        cdf['proba'] = cdf['proba'] / cdf['proba'].max() # 20.7 PPV
    #     results[combination_id] = cdf

    combinations = []
    for i in range(1, len(results) + 1):
        combinations += itertools.combinations(results.keys(), i)

    combined_results = []

    performance_by_threshold = []

    for combination in tqdm(combinations):
        # Take the majority vote of the models in the combination
        votes = np.zeros(results[combination[0]].shape[0])
        for combination_id in combination:
            votes += results[combination_id]["proba"]

        votes = votes / len(combination)
        current = results[combination[0]].copy()

        tuning = True
        threshold = 0.49
        step = 0.01
        if tuning:
            while True:
                predicted = votes >= threshold + step
                tp = current[(predicted == True) & (current["actual"] == True)].shape[0]
                fp = current[(predicted == True) & (current["actual"] == False)].shape[0]
                fn = current[(predicted == False) & (current["actual"] == True)].shape[0]
                try:
                    ppv = tp / (tp + fp)
                    fnr = fn / (tp + fn)
                except ZeroDivisionError:
                    break
                                    
                if fnr > 0.0:
                    break
                else:
                    threshold += step
                    current["predicted"] = votes >= threshold
                

        current["predicted"] = votes >= threshold

        # Calculate the PPV and FNR
        tp = current[(current["predicted"] == True) & (current["actual"] == True)].shape[0]
        fp = current[(current["predicted"] == True) & (current["actual"] == False)].shape[0]
        fn = current[(current["predicted"] == False) & (current["actual"] == True)].shape[0]
        ppv = tp / (tp + fp)
        fnr = fn / (tp + fn)

        combined_results.append((combination, len(combination), ppv, fnr, threshold))

    combined_results = pd.DataFrame(combined_results, columns=["combination", "model_count", "ppv", "fnr", "threshold"])
    combined_results.sort_values("ppv", ascending=False).head(100).to_csv("combined_results_majority_probabilities_norm.csv", index=False)
    combined_results.sort_values("ppv", ascending=False).head

    # Plot the results
    # plt.figure(figsize=(7, 4), dpi=500)
    # sns.lineplot(data=combined_results, x='model_count', y='ppv', marker='o', label='Combined Predictions')
    # plt.xlabel("Model count")
    # plt.ylabel("PPV")
    # plt.ylim(0.08, 0.22)
    # plt.axhline(0.105, color='red', linestyle='--', label='Current NSO PPV')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('unanimous_voting_improvements.png')
    # plt.savefig('unanimous_voting_improvements.eps', dpi=1000, format='eps')
    # plt.close()
    return combined_results


def plot_agreements(df: pd.DataFrame) -> None:
    results = [(combination_id, df[["case_index", "actual", "predicted", "proba"]]) for combination_id, df in df.groupby("combination_id")]
    results = [(combination_id, df.sort_values("case_index").reset_index()) for combination_id, df in results if df.shape == (616910, 4)]
    results = dict(results)

    if 368 in results:
        results.__delitem__(368)

    # For each combination filter out the cases that were predicted as positive
    case_list = set()
    for combination_id, result in results.items():
        result["predicted"] = result["predicted"].astype(bool)
        results[combination_id] = result[result["predicted"] == True]
        # results[combination_id] = results[combination_id][results[combination_id]['actual'] == True]
        for case_index in result[result["predicted"] == True]["case_index"]:
            case_list.add(case_index)

    case_list = list(case_list)

    # Create a heatmap of the cases that were predicted as positive by the different models
    heatmap = np.zeros((len(results), len(case_list)))
    for i, (combination_id, result) in enumerate(results.items()):
        for j, case_index in enumerate(case_list):
            if case_index in result["case_index"].values:
                # If true positive
                heatmap[i, j] = result[result["case_index"] == case_index]["proba"].values[0]
            else:
                heatmap[i, j] = None

    # Grab indices of true positives for a single row
    true_positives = results[340][results[340]["actual"] == True]["case_index"]
    heatmap = pd.DataFrame(heatmap, index=[str(cid) for cid in results.keys()], columns=case_list)
    heatmap = heatmap.dropna(axis=1, how="all")
    heatmap = heatmap.astype(float)

    # Cases along X-axis by amount of agreement (cumulative probability)
    heatmap = pd.concat([heatmap.T, heatmap.sum(axis=0).sort_values(ascending=False)], axis=1).T
    heatmap = heatmap.rename(index={0: "Total Agreement"})
    heatmap = heatmap.sort_values("Total Agreement", axis=1, ascending=False)

    # Order the position of the rows by the number of null values
    heatmap["null_count"] = heatmap.isnull().sum(axis=1)
    heatmap = heatmap.sort_values("null_count")
    heatmap = heatmap.drop(columns="null_count")

    # Add the true positives to the top of the heatmap
    new = pd.DataFrame(np.zeros((1, heatmap.shape[1])), columns=heatmap.columns)
    new.loc[0, :] = None
    new.loc[0, true_positives] = 1 + 1 / 256  # Set all true positives to Pink (1 + 1/256)
    heatmap = pd.concat([heatmap, new])
    heatmap.rename(index={0: "True Positives"}, inplace=True)

    heatmap = heatmap.T
    heatmap["Total Agreement"] = heatmap["Total Agreement"] / heatmap["Total Agreement"].max()
    heatmap["Total Agreement"] = heatmap["Total Agreement"] / 2 + 0.5
    heatmap = heatmap.T

    # Create a heatmap color map with pink for true positives
    viridis = cm.get_cmap("viridis", 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    newcolors[-1:, :] = pink
    newcmp = ListedColormap(newcolors)

    plt.figure(figsize=(30, 10), dpi=500)
    sns.heatmap(heatmap, cmap=newcmp, cbar_kws={"label": "Predicted Probability"})
    plt.ylabel("Combination ID")
    plt.xticks([])
    plt.xlabel("Case index")
    # plt.title('Probabilities For Cases Predicted As Positive Sorted by Total Agreement')
    plt.tight_layout()
    plt.savefig("cases_sorted_by_agreement_and_nulls.png", dpi=500)
    plt.show()
    plt.close()

    # Sort by true positive
    heatmap = heatmap.sort_values(["True Positives", "Total Agreement"], axis=1, ascending=False)
    heatmap = heatmap.T
    heatmap["Total Agreement"] = heatmap["Total Agreement"] / heatmap["Total Agreement"].max()
    heatmap["Total Agreement"] = heatmap["Total Agreement"] / 2 + 0.5
    heatmap = heatmap.T

    plt.figure(figsize=(30, 10), dpi=500)
    sns.heatmap(heatmap, cmap=newcmp, cbar_kws={"label": "Predicted Probability"})
    plt.ylabel("Combination ID")
    plt.xticks([])
    plt.xlabel("Case index")
    # plt.title('Probabilities For Cases Predicted As Positive Sorted by Total Agreement (True Positives First)')
    plt.tight_layout()
    plt.savefig("cases_sorted_by_agreement_and_nulls_tp.png", dpi=500, format="png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("./notebooks/general/direct_results/original_data_results.csv")
    df = df.drop(
        columns=[
            "Unnamed: 0",
            "record_id.1",
            "record_id.2",
            "id.1",
            "id.2",
            "id.3",
            "id.4",
            "model_file",
            "result_file",
            "metadata",
            "created_at",
            "latest",
        ]
    )
    df = df.rename(columns={"index": "case_index"})
    # print(df.columns)
    df = df[df["fnr"] == 0.0]
    #    df = df.sort_values(['case_index', 'ppv'], ascending=[True, False])
    #    df = df[df['combination_id'].isin([428, 447, 639, 394, 397, 383, 404, 423, 340, 371])]
    # Take top 20 combinations
    best_ids = df.groupby("combination_id")["ppv"].mean().sort_values(ascending=False).head(10).index
    #    print(best_ids)
    df = df[df["combination_id"].isin(best_ids)]

    print(f"Starting with {len(best_ids)} combinations!")
    best_and_combinations = (
        and_combination(df).sort_values(["ppv", "model_count", "fnr"], ascending=[False, True, True]).groupby("model_count").max().reset_index()
    )
    best_prob_combinations = (
        simple_vote(df).sort_values(["ppv", "model_count", "fnr"], ascending=[False, True, True]).groupby("model_count").max().reset_index()
    )

    print(best_and_combinations)
    print(best_prob_combinations)

    # simple_vote(df)
    plt.figure(figsize=(7, 4), dpi=500)
    sns.lineplot(data=best_and_combinations, x="model_count", y="ppv", marker="o", label="Unanimous Voting", errorbar=None)
    # sns.lineplot(data=best_prob_combinations, x="model_count", y="ppv", marker="o", label="Probabilistic Voting", errorbar=None)
    plt.xlabel("Model count")
    plt.ylabel("PPV")
    plt.ylim(0.08, 0.22)
    plt.axhline(0.105, color="red", linestyle="--", label="Current NSO PPV")
    plt.axvline(x=6, ymin=0.6, ymax=0.8, color="green", linestyle="--", label="Ideal Model Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("overlap_improvements.png")
    plt.savefig("overlap_improvements.eps", dpi=1000, format="eps")
