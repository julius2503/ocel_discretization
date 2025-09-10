import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run_discretization(ocel, attribute, values, ids):
    algorithm = attribute["algorithm"]

    match algorithm["name"]:
        case "equal-freq":
            bins = int(algorithm["parameters"]["bins"])
            return perform_equal_frequency_binning(values, bins)
        case "equal-width":
            bins = int(algorithm["parameters"]["bins"])
            return perform_equal_width_binning(values, bins)
        case "chi-merge":
            labels = algorithm["parameters"]["labels"]
            max_intervals = int(algorithm["parameters"]["max_interval"])
            significance = float(algorithm["parameters"]["significance"])
            return perform_chi_merge_binning(ocel, attribute, ids, labels, max_intervals, significance)
        case "k-means":
            labels = algorithm["parameters"]["labels"]
            n_clusters = int(algorithm["parameters"]["clusters"])
            return perform_kmeans_clustering(ocel, attribute, ids, labels, n_clusters)

def perform_equal_frequency_binning(values: List[float], n_bins: int) -> List[Tuple[float, float]]:
    if not values or n_bins <= 0:
        return []

    sorted_values = sorted(values)
    n = len(sorted_values)
    bins = min(n_bins, n)

    target_freq = n / bins
    intervals = []
    start_idx = 0

    for i in range(bins - 1):
        end_idx = int((i + 1) * target_freq) - 1
        end_idx = min(end_idx, n - 1)

        while (end_idx < n - 1 and sorted_values[end_idx] == sorted_values[end_idx + 1]):
            end_idx += 1

        start_val = sorted_values[start_idx]
        end_val = sorted_values[end_idx]

        intervals.append((start_val, end_val))
        start_idx = end_idx + 1

    if start_idx < n:
        intervals.append((sorted_values[start_idx], sorted_values[-1]))

    return intervals

def perform_equal_width_binning(values: List[float], n_bins: int) -> List[Tuple[float, float]]:
    if not values or n_bins <= 0:
        return []

    arr = np.array(values, dtype=float)
    min_val, max_val = arr.min(), arr.max()

    width = (max_val - min_val) / n_bins
    intervals: List[Tuple[float, float]] = []

    for i in range(n_bins):
        start = round(min_val + i * width, 2)
        end = round(min_val + (i + 1) * width, 2) if i < n_bins - 1 else round(max_val, 2)
        intervals.append((start, end))

    non_overlap: List[Tuple[float, float]] = []
    for i, (start, end) in enumerate(intervals):
        # Wenn nicht das letzte Intervall und end == nächster Start:
        if i < len(intervals) - 1 and end == intervals[i + 1][0]:
            end = end - 0.01
        non_overlap.append((start, round(end, 2)))
    return non_overlap


def perform_chi_merge_binning(ocel, attribute, ids, labels, max_intervals, significance_level):
    attr = attribute["attribute"]
    label = json.loads(labels[0])["attribute"]
    mapping = _map_attribute_label(ocel, ids, attribute["attribute"], attribute["type"], attribute["qualifier"], labels)
    mapping = mapping.sort_values(attr)
    counts = mapping.groupby(attr)[label].value_counts().unstack(fill_value=0)

    intervals = [
            (val, val, counts.loc[val])
            for val in counts.index
        ]

    categories = counts.columns.tolist()
    dfree = len(categories) - 1
    threshold = chi2.ppf(1 - significance_level, dfree) if dfree > 0 else 0.0

    while len(intervals) > max_intervals:
        chi_vals = [
            _chi2_stat(intervals[i][2], intervals[i+1][2])
            for i in range(len(intervals) - 1)
        ]
        min_chi = min(chi_vals)
        idx = chi_vals.index(min_chi)

        if min_chi > threshold:
            break

        start1, _, counts1 = intervals[idx]
        _, end2, counts2 = intervals[idx + 1]
        merged_counts = counts1 + counts2
        intervals[idx:idx+2] = [(start1, end2, merged_counts)]

    return [(start, end) for start, end, _ in intervals]

def perform_kmeans_clustering(ocel, attribute, ids, labels, n_clusters):
    attr = attribute["attribute"]
    label_names = [json.loads(lbl)["attribute"] for lbl in labels]
    mapping = _map_attribute_label(ocel, ids, attribute["attribute"], attribute["type"], attribute["qualifier"], labels)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(mapping[[attr]])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(mapping[label_names])

    X = np.hstack([X_num, X_cat])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(X)

    df_result = mapping.copy()
    df_result["cluster"] = cluster_ids

    cluster_values: Dict[int, List[float]] = {}
    for cid in range(n_clusters):
        vals = df_result.loc[df_result["cluster"] == cid, attr].unique().tolist()
        cluster_values[cid] = vals

    intervals: List[Tuple[float, float]] = []

    for key, values in cluster_values.items():
        min_val = min(values)
        max_val = max(values)

        intervals.append((min_val, max_val))

    return intervals

def _map_attribute_label(ocel, ids, attribute, type, qualifier, labels):
    match type:
        case "EVENT":
            df_filtered = ocel.events.loc[ocel.events["ocel:eid"].isin(ids), ["ocel:eid", attribute]].copy()
            for label in labels:
                label = json.loads(label)

                match label["type"]:
                    case "EVENT":
                        df_label = ocel.events.loc[:, ["ocel:eid", label["attribute"]]]
                        df_filtered = df_filtered.merge(df_label, on="ocel:eid", how="left")

                    case "OBJECT":
                        rel = ocel.relations[["ocel:eid", "ocel:oid"]]
                        obj = ocel.objects[["ocel:oid", "ocel:type", label["attribute"]]]
                        merged = rel.merge(obj, on="ocel:oid", how="left")
                        merged = merged[merged["ocel:type"] == label["qualifier"]]
                        agg = merged.groupby("ocel:eid")[label["attribute"]].apply(_most_common)
                        df_filtered = df_filtered.merge(agg.reset_index(), on="ocel:eid", how="left")

                    case _:
                        raise Exception("Label ist weder vom Typ Event noch Object")

            return df_filtered.drop("ocel:eid", axis=1)

        case "OBJECT":
            df_filtered = ocel.objects.loc[ocel.objects["ocel:oid"].isin(ids), ["ocel:oid", attribute]].copy()
            for label in labels:
                label = json.loads(label)
                df_label = ocel.objects.loc[:, ["ocel:oid", label["attribute"]]]
                df_filtered = df_filtered.merge(df_label, on="ocel:oid", how="left")

            return df_filtered.drop("ocel:oid", axis=1)

        case _:
            raise Exception("Numerisches Attribut ist weder vom Typ Event noch Object")

def _most_common(series: pd.Series):
    counts = series.dropna().value_counts()
    return counts.idxmax() if not counts.empty else None

def _chi2_stat(c1: pd.Series, c2: pd.Series) -> float:
    obs = np.vstack([c1.to_numpy(dtype=float), c2.to_numpy(dtype=float)])
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    exp = row_sum.dot(col_sum) / total
    mask = exp > 0
    return float(((obs - exp) ** 2 / exp)[mask].sum())
