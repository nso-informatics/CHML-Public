from runtime.analysis import *
from scripts.feature_sets import *
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import List, Union
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")


def load_data(f: Filter) -> pd.DataFrame:
    original = load_chml(include_episode=True).reset_index()
    original = original.drop(columns=["tsh_squared", "tsh_cubed", "weight_to_gestational_age", "weight_to_age_at_collection"])
    a = Analysis(records_path=Path("/data/CHML/records"), load_dataframes=False, filter=f)
    a.save_analysis_db()
    data = pd.read_csv(a.analytics_file)
#    if len(data) == 0:
#        raise ValueError("No records found")
#    if len(data) != 5:
#        raise ValueError("Narrow the filter to a single combination")
    return original, data # type: ignore


def load_results(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 0:
        return pd.DataFrame()
    data["results"] = data.apply(lambda x: pd.read_csv(x["result_file"]), axis=1)
    results = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        result_df = row["results"]
        row_data = row.drop("results")
        result_df[row_data.index] = row_data
        result_df["model"] = pickle.load(open(row_data["model_file"], "rb")) # type: ignore
        result_df["fold"] = row_data["fold"]
        results.append(result_df)
    return pd.concat(results)


def generate_explanations(output: pd.DataFrame, original: pd.DataFrame, cases: List[int]):
    output.index = output["index"].astype(int) # type: ignore
    output.sort_index(inplace=True)
    output = pd.merge(output, original, right_index=True, left_index=True, how="left")
    focus = output[output["index_x"].isin(cases)]
    shap_values = []
    
    os.makedirs("/data/CHML/explanations", exist_ok=True)
    os.makedirs("/data/CHML/explanations/waterfall", exist_ok=True)
    for name, group  in focus.groupby(["tag", "combination_id"]):
        tag, combination_id = name
        for _, row in focus.iterrows():
            print(f"Generating explanation for case {row['episode']}")
            model = row["model"]
            fold = row["fold"]
            index = row["index_x"]
            episode = row["episode"]
            X_train_i = output[output["fold"] != fold].index
            X_train = original.loc[original.index.isin(X_train_i)].drop(columns=["definitive_diagnosis", "index", "episode"])
            explainer = shap.Explainer(model.predict_proba, X_train) # type: ignore
            X_test = original.loc[original.index == index].drop(columns=["definitive_diagnosis", "index", "episode"])
            shap_values = explainer(X_test)
            shap_values = combine_one_hot(shap_values, "Sex", ['sex' in x for x in shap_values.feature_names], False)
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            #plt.title(f"Case {row['episode']},\n tag={row['tag']}, combination={row['combination_id']}\n Definitive Diagnosis = {bool(original.loc[original.index == index].iloc[0]['definitive_diagnosis'])} \n Prediction = {bool(model.predict(X_test)[0])}")
            plt.tight_layout()
            plt.savefig(f"/data/CHML/explanations/waterfall/{episode}_{tag}_{combination_id}_{fold}.eps", dpi=500, format="eps")
            plt.close()


def remove_feature_from_shap_values(shap_values, feature_to_remove):
    feature_index_to_remove = shap_values.feature_names.index(feature_to_remove)
    filtered_shap_values = np.delete(shap_values.values, feature_index_to_remove, axis=1)
    filtered_feature_names = np.delete(shap_values.feature_names, feature_index_to_remove)
    shap_values_data = np.delete(shap_values.data, feature_index_to_remove, axis=1)
    return shap.Explanation(values=filtered_shap_values, base_values=shap_values.base_values, data=shap_values_data, feature_names=filtered_feature_names)


def combine_one_hot(shap_values, name, mask, return_original=True):
    """  shap_values: an Explanation object
          name: name of new feature
          mask: bool array same lenght as features

         This function assumes that shap_values[:, mask] make up a one-hot-encoded feature
    """
    mask = np.array(mask)
    mask_col_names = np.array(shap_values.feature_names, dtype='object')[mask]

    #print(shap_values)
    #print(shap_values.shape)

    sv_name = shap.Explanation(shap_values.values[:, mask],
                               feature_names=list(mask_col_names),
                               data=shap_values.data[:, mask],
                               base_values=shap_values.base_values,
                               display_data=shap_values.display_data,
                               instance_names=shap_values.instance_names,
                               output_names=shap_values.output_names,
                               output_indexes=shap_values.output_indexes,
                               lower_bounds=shap_values.lower_bounds,
                               upper_bounds=shap_values.upper_bounds,
                               main_effects=shap_values.main_effects,
                               hierarchical_values=shap_values.hierarchical_values,
                               clustering=shap_values.clustering,
                               )

    new_data = (sv_name.data * np.arange(sum(mask))).sum(axis=1).astype(int)

    svdata = np.concatenate([
        shap_values.data[:, ~mask],
        new_data.reshape(-1, 1)
    ], axis=1)

    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]

    svdisplay_data = np.concatenate([
        svdd,
        mask_col_names[new_data].reshape(-1, 1)
    ], axis=1)

    new_values = sv_name.values.sum(axis=1)
    svvalues = np.concatenate([
        shap_values.values[:, ~mask],
        new_values.reshape(-1, 1)
    ], axis=1)
    svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

    sv = shap.Explanation(svvalues,
                          base_values=shap_values.base_values,
                          data=svdata,
                          display_data=svdisplay_data,
                          instance_names=shap_values.instance_names,
                          feature_names=svfeature_names,
                          output_names=shap_values.output_names,
                          output_indexes=shap_values.output_indexes,
                          lower_bounds=shap_values.lower_bounds,
                          upper_bounds=shap_values.upper_bounds,
                          main_effects=shap_values.main_effects,
                          hierarchical_values=shap_values.hierarchical_values,
                          clustering=shap_values.clustering,
                          )
    if return_original:
        return sv, sv_name
    else:
        return sv


def sum_featues_from_shap_values(shap_values, features_to_combine, new_feature_name):
    feature_indices = [shap_values.feature_names.index(feature) for feature in features_to_combine]

    #print(f"Combined Source: {shap_values.values[:, feature_indices]}")
    #print(f"Data: {shap_values.data[:, feature_indices]}")
    exit(0)
    
    combined_values = np.sum(shap_values.values[:, feature_indices], axis=1)
    feature_names = np.delete(shap_values.feature_names, feature_indices)
    feature_names = np.append(feature_names, new_feature_name)
    data = np.delete(shap_values.data, feature_indices, axis=1)
    data = np.append(data, np.sum(shap_values.data[:, feature_indices], axis=1).reshape(-1, 1), axis=1)
    values = np.delete(shap_values.values, feature_indices, axis=1)
    values = np.append(values, combined_values.reshape(-1, 1), axis=1)

    print(f"Feature names: {feature_names}")
    print(f"Data: {data}")
    print(f"Values: {values}")
    print(f"Base Values: {shap_values.base_values}")

    print(f"Shap Values: {shap_values}")
    print(f"Shap Values Shape: {shap_values.values.shape}")
    print(f"Data Shape: {data.shape}") 


    return shap.Explanation(values=values, base_values=shap_values.base_values, data=data, feature_names=feature_names)


def generate_shap_summaries(output: pd.DataFrame, original: pd.DataFrame):
    if len(output) == 0:    
        return
    output.index = output["index"].astype(int) # type: ignore
    output.sort_index(inplace=True)
    output = pd.merge(output, original, right_index=True, left_index=True, how="left")
    # original = original[original["definitive_diagnosis"] == 1]
    # Undersample the negative cases
    original = pd.concat([original[original["definitive_diagnosis"] == 1], original[original["definitive_diagnosis"] == 0].sample(n=100)])
    os.makedirs("/data/CHML/explanations", exist_ok=True)

    for name, group  in output.groupby(["tag", "combination_id"]):
        tag, combination_id = name
        print(f"Generating explanation for model {name}")
        model = group["model"].iloc[0]
        fold = group["fold"].iloc[0]

        # Choose the train and test sets based on the fold indicies
        X_train_i = output[output["fold"] != fold].index
        X_train = original.loc[original.index.isin(X_train_i)].drop(columns=["definitive_diagnosis", "index", "episode"])
        X_test = original[~original.index.isin(X_train_i)].drop(columns=["definitive_diagnosis", "index", "episode"])
        print(f"Train Shape: {X_train.shape}")
        print(f"Test Shape: {X_test.shape}")
        
        try:
            importances = pd.DataFrame(model.model.best_estimator_.feature_importances_, index=X_train.columns, columns=["importance"])
            importances = importances.sort_values(by="importance", ascending=False)
            importances.to_csv(f"/data/CHML/explanations/{tag}_{combination_id}_importances.csv")
            importances.index = [x.replace("age_at_collection", "Age at Collection") for x in importances.index]
            importances = importances.head(15)
            importances = importances[importances["importance"] > 0.0]
            plt.figure(dpi=500)
            sns.barplot(x=importances.index, y=importances["importance"], palette="Set3")
            #plt.title(f"Model {name}")
            plt.xlabel("Feature")
            plt.xticks(rotation=90)
            plt.ylabel("Importance")
            plt.legend().remove()
            plt.tight_layout()
            plt.savefig(f"/data/CHML/explanations/{tag}_{combination_id}_importances.eps", dpi=500, format="eps")
            plt.close()
        except AttributeError:
            print("Model does not have feature importances")

        explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 0], X_train, feature_names=X_train.columns) # type: ignore
        #shap_values = explainer(X_test.sample(n=10))
        shap_values = explainer(X_test)
        
        # shap_values = sum_featues_from_shap_values(shap_values, ['sex_male', 'sex_female'], "sex")
        shap_values = combine_one_hot(shap_values, "Sex", ['sex' in x for x in shap_values.feature_names], False)
        shap.plots.beeswarm(shap_values, show=False, cluster_threshold=0.5, max_display=20, order=shap.Explanation.abs.mean(0))
        #shap.plots.waterfall(shap_values[0, :, 1], show=False)
        # plt.title(f"Model {name}")
        plt.tight_layout()
        plt.savefig(f"/data/CHML/explanations/{tag}_{combination_id}.eps", dpi=500, format="eps")
        plt.close()

        # Feature importances from model
        # Remove TSH from shap_value
        #for feature_to_remove in ['TSH']:
        #for feature_to_remove in ['tsh_squared', 'tsh_cubed', 'TSH']:
        #    shap_values = remove_feature_from_shap_values(shap_values, feature_to_remove)
        
        shap.plots.beeswarm(shap_values, show=False, cluster_threshold=0.5, max_display=20, order=shap.Explanation.abs.mean(0))
        #plt.title(f"Model {name}")
        plt.tight_layout()
        plt.savefig(f"/data/CHML/explanations/{name[0]}_{name[1]}_no_tsh.eps", dpi=500, format="eps")
        plt.close()
        return shap_values

def explain_cases(f: Filter, cases: Union[int, List[int]]):
    if isinstance(cases, int):
        cases = [cases]

    original, data = load_data(f)
    case_results = load_results(data)
    generate_explanations(case_results, original, cases)

def explain_models(f: Filter):
    original, data = load_data(f)
    case_results = load_results(data)
    shap_values = generate_shap_summaries(case_results, original)    
    return shap_values

if __name__ == "__main__":
    data = load_chml(include_episode=True)
    data = data.reset_index()
    data = data.drop(columns=["tsh_squared", "tsh_cubed", "weight_to_gestational_age", "weight_to_age_at_collection"])
    #for combination_id in [449, 452, 286, 365, 425, 434, 33, 439, 85, 38]:
#    for combination_id in [428, 447, 639, 394, 397, 383, 404, 423, 340, 371]:
    #f = Filter(tag="original_data$", model="(?!^Bagging$)(^.*$)")
    f = Filter(tag="original_data$", model="Forest")
    a = Analysis(records_path=Path("/data/CHML/records"), load_dataframes=False, filter=f)
    a.save_analysis_db()
    data = pd.read_csv(a.analytics_file)
    shap_values = []
    data = data[['tag', 'combination_id', 'ppv', 'fnr']].groupby(['tag', 'combination_id']).mean().reset_index()
    data = data[data['fnr'] == 0.0]
    data = data.sort_values(by='ppv', ascending=False)[:10]
    print(data)
    for combination_id in data["combination_id"].unique():
    #for combination_id in [85, 38]:
        f = Filter(tag="original_data$", combination_id=combination_id)
        #cases = [259402, 14636] + data[data['definitive_diagnosis'] == 1].sample(n=10).index.tolist()
        #print(cases)
        # explain_cases(f, cases=cases)
        shap_values += [explain_models(f)]
    feature_names = shap_values[0].feature_names
    print(shap_values[0])

    impact = np.zeros(len(feature_names))

    # Enumerate and loop over feature names
    for i, feature in enumerate(feature_names):
        #impact[i] = sum([np.abs(sv.values[:, i]). for sv in shap_values])
        for sv in shap_values:
            cutoff = np.percentile(np.abs(sv.values[:, i]), 90)
            arr = np.abs(sv.values[:, i])
            impact[i] += np.sum(arr[arr > cutoff])
   
    # Put into a pandas series
    impact = pd.Series(impact, index=feature_names)
    impact = impact.sort_values(ascending=False)
    print(impact)
    impact.to_csv("/data/CHML/explanations/impact_nonbagging.csv")
