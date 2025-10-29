import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pm4py import OCEL
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run_discretization(
    ocel: OCEL, attribute: Dict[str, Any], values: List[float], ids: List[Any]
) -> Tuple[List[Tuple[float, float]], Dict[Any, int] | None]:
    """
    Dispatch to the selected discretization algorithm.

    Inputs:
        ocel      – OCEL instance (for chi-merge)
        attribute – descriptor dict with:
                    {
                            "attribute": str,
                            "type": str // "EVENT" or "OBJECT",
                            "qualifier": str,
                            "algorithm": {
                                "name": srt // "equal-freq" or "equal-width" or "chi-merge" or "k-means",
                                "parameters": {}
                            }
                    }
        values    – list of numeric values for this attribute
        ids       – list of event or object IDs corresponding to values

    Output:
        [
            (start: float, end: float)
            eid-cluster mapping
        ]
    """
    algo = attribute["algorithm"]["name"]
    params = attribute["algorithm"]["parameters"]

    if algo == "equal-freq":
        bins = int(params.get("bins", 5))
        return perform_equal_frequency_binning(values, bins), None

    if algo == "equal-width":
        bins = int(params.get("bins", 5))
        return perform_equal_width_binning(values, bins), None

    if algo == "chi-merge":
        labels = params.get("labels", {})
        max_int = int(params.get("max_interval", 5))
        signif = float(params.get("significance", 0.05))
        return perform_chi_merge_binning(ocel, attribute, ids, labels, max_int, signif), None

    if algo == "k-means":
        labels = params.get("labels", {})
        n_clusters = int(params.get("clusters", 3))
        return perform_kmeans_clustering(ocel, values, attribute, ids, labels, n_clusters)

    raise ValueError(f"Unknown discretization algorithm: {algo}")


def perform_equal_frequency_binning(values: List[float], n_bins: int) -> List[Tuple[float, float]]:
    """Equal-frequency binning."""
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

        while end_idx < n - 1 and sorted_values[end_idx] == sorted_values[end_idx + 1]:
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
        intervals.append((float(start), float(end)))

    non_overlap: List[Tuple[float, float]] = []
    for i, (start, end) in enumerate(intervals):
        if i < len(intervals) - 1 and end == intervals[i + 1][0]:
            end = end - 0.01
        non_overlap.append((float(start), round(float(end), 2)))
    return non_overlap


def perform_chi_merge_binning(
    ocel: OCEL, attribute: Dict[str, Any], ids: List[Any], labels: List[str], max_intervals: int, significance: float
) -> List[Tuple[float, float]]:
    """ChiMerge algorithm merging adjacent bins by chi-square test."""
    numeric_attr = attribute.get("attribute", "")

    df = ocel.events if attribute.get("type") == "EVENT" or attribute.get("aggregation", "") else ocel.objects
    for col in [df.columns[0], numeric_attr]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame")

    mapping = _map_attribute_label(ocel, ids, attribute, labels)
    mapping["combined"] = mapping[[json.loads(label).get("attribute", "") for label in labels]].astype(str).agg("_".join, axis=1)
    if numeric_attr not in mapping.columns:
        raise KeyError("Numeric missing after mapping")

    counts = mapping.groupby(numeric_attr)["combined"].value_counts().unstack(fill_value=0).sort_index()

    intervals = [(val, val, counts.loc[val]) for val in counts.index]

    categories = counts.columns.tolist()
    dfree = len(categories) - 1

    while len(intervals) > max_intervals:
        threshold = chi2.ppf(1 - significance, dfree) if dfree > 0 else 0.0

        chi_vals = [_chi2_stat(intervals[i][2], intervals[i + 1][2]) for i in range(len(intervals) - 1)]

        min_chi = min(chi_vals)
        if min_chi > threshold:
            break

        idx = chi_vals.index(min_chi)
        start1, _, counts1 = intervals[idx]
        _, end2, counts2 = intervals[idx + 1]
        merged_counts = counts1.add(counts2, fill_value=0)

        intervals[idx : idx + 2] = [(start1, end2, merged_counts)]

    while len(intervals) > max_intervals:
        chi_vals = [_chi2_stat(intervals[i][2], intervals[i + 1][2]) for i in range(len(intervals) - 1)]

        idx = chi_vals.index(min(chi_vals))
        start1, _, counts1 = intervals[idx]
        _, end2, counts2 = intervals[idx + 1]
        merged_counts = counts1.add(counts2, fill_value=0)

        intervals[idx : idx + 2] = [(start1, end2, merged_counts)]

    return [(start, end) for start, end, _ in intervals]


def perform_kmeans_clustering(
    ocel: OCEL, values: List[float], attribute: Dict[str, Any], ids: List[Any], labels: List[str], n_clusters: int
) -> Tuple[List[Tuple[float, float]], Dict[Any, int] | None]:
    X_num = np.array(values, dtype=float).reshape(-1, 1)

    if labels:
        df_labels = _map_attribute_label(ocel, ids, attribute, labels)
        label_cols = [col for col in df_labels.columns if col != attribute.get("attribute", "")]
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_label = encoder.fit_transform(df_labels[label_cols])
        X = np.hstack([X_num, X_label])
    else:
        X = X_num

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    cluster_info = []
    for cid in range(n_clusters):
        subset = X_num[cluster_labels == cid].flatten()
        if subset.size == 0:
            cluster_info.append({"original_cluster_id": cid, "min": float("nan"), "max": float("nan")})
        else:
            cluster_info.append({"original_cluster_id": cid, "min": float(subset.min()), "max": float(subset.max())})

    cluster_info = sorted(cluster_info, key=lambda x: x["min"])

    original_to_sorted = {info["original_cluster_id"]: i for i, info in enumerate(cluster_info)}

    cluster_labels = np.array([original_to_sorted[cl] for cl in cluster_labels])

    intervals: List[Tuple[float, float]] = [(info["min"], info["max"]) for info in cluster_info]

    id_to_cluster = {eid: int(cl) for eid, cl in zip(ids, cluster_labels)}

    return intervals, id_to_cluster if labels else None


def _map_attribute_label(ocel: OCEL, ids: List[Any], attribute: Dict[str, Any], labels: List[str]) -> pd.DataFrame:
    """
    Build DataFrame with original numeric attribute and associated label.
    """
    df = ocel.events if attribute.get("type", "") == "EVENT" or attribute.get("aggregation", "") else ocel.objects
    df = df.loc[df[df.columns[0]].isin(ids), ["ocel:eid", attribute.get("attribute", "")]].copy()

    for label in labels:
        label = json.loads(label)
        qualifier = label.get("qualifier")
        if label.get("type", "") == "EVENT":
            col = ocel.events.loc[ocel.events["ocel:activity"] == qualifier, ["ocel:eid", label.get("attribute", "")]]
            df = df.merge(col, on="ocel:eid", how="left")
        else:
            rel = ocel.relations.loc[ocel.relations["ocel:type"] == qualifier, ["ocel:eid", "ocel:oid"]]
            obj = ocel.objects.loc[ocel.objects["ocel:type"] == qualifier, ["ocel:oid", label.get("attribute", "")]]
            df = df.merge(
                rel.merge(obj, on="ocel:oid", how="left").groupby("ocel:eid")[label.get("attribute", "")].apply(_most_common),
                on="ocel:eid",
                how="left",
            )

    return df.drop("ocel:eid", axis=1)


def _most_common(series: pd.Series):
    """
    Return the most frequent value.
    """
    counts = series.dropna().value_counts()
    return counts.idxmax() if not counts.empty else None


def _chi2_stat(c1: pd.Series, c2: pd.Series) -> float:
    """Compute chi-square statistic between two count vectors."""
    obs = np.vstack([c1.to_numpy(dtype=float), c2.to_numpy(dtype=float)])
    total = obs.sum()
    if total == 0:
        return 0.0

    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    exp = row_sum.dot(col_sum) / total

    mask = exp > 0
    chi2_vals = np.zeros_like(obs)
    chi2_vals[mask] = (obs[mask] - exp[mask]) ** 2 / exp[mask]

    return float(chi2_vals.sum())
