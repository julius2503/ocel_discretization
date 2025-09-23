import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pm4py import OCEL

from config import DATA_FOLDER
from src.discretization import run_discretization
from src.preprocessing import load_ocel, save_ocel, split_numerical_attribute


def run_itemize(ocel: OCEL, attribute: Dict[str, Any]) -> List[Dict]:
    """
    Convert a selected attribute into discrete items for mining.

    Inputs:
        ocel: OCEL
        attribute: {
            "attribute": str,
            "type": str // "EVENT" or "OBJECT",
            "qualifier": str,
            "numeric": bool,
            "split_attributes": List[Dict],
            "aggregation": str
        }
    Output:
        [{
            "attribute": str,
            "type": str // "EVENT" or "OBJECT",
            "qualifier": str,
            "aggregate": str,
            "value": str // for categorical attributes,
            "interval": {
                "start": float,
                "end": float
            } // for discretized attributes,
            "mapping": [str] // eids/oids for this item
        }]

    """
    attr = attribute.get("attribute", "")
    qualifier = attribute.get("qualifier", "")
    type = attribute.get("type", "")
    splits = attribute.get("split_attributes", [])
    aggregation = attribute.get("aggregation", "")

    if attribute.get("numeric", False):
        if type == "EVENT":
            df = split_numerical_attribute(ocel, type, splits)

            values = df[df["ocel:activity"] == qualifier][attr].astype(float).tolist()
            ids = df[df["ocel:activity"] == qualifier]["ocel:eid"].tolist()

        elif type == "OBJECT":
            df = split_numerical_attribute(ocel, type, splits)
            if aggregation:
                df = ocel.events
                values = df[attr].astype(float).tolist()
                ids = []
            else:
                values = df[df["ocel:type"] == qualifier][attr].astype(float).tolist()
                ids = df[df["ocel:type"] == qualifier]["ocel:oid"].tolist()
        else:
            raise ValueError(f"Unknown type for numeric attribute: {type}")

        intervals, id_to_cluster = run_discretization(ocel, attribute, values, ids)
        if not aggregation:
            _apply_intervals_to_ocel(attribute, intervals, id_to_cluster)

        cluster_to_ids: dict[int, list[Any]] = {}
        if id_to_cluster:
            for eid, cid in id_to_cluster.items():
                cluster_to_ids.setdefault(cid, []).append(eid)


        return [
            {
                "attribute": attr,
                "type": type,
                "qualifier": qualifier,
                "interval": {"start": start, "end": end},
                "aggregate": aggregation,
                "mapping": cluster_to_ids.get(idx, [])
            }
            for idx, (start, end) in enumerate(intervals)
        ]

    if type == "EVENT":
        values = ocel.events.loc[
            ocel.events["ocel:activity"] == qualifier, attr
        ].astype(str).unique()
    elif type == "OBJECT":
        values = ocel.objects.loc[
            ocel.objects["ocel:type"] == qualifier, attr
        ].astype(str).unique()
    else:
        raise ValueError(f"Unknown type for categorical attribute: {type}")

    return [
        {"attribute": attr, "type": type, "qualifier": qualifier, "value": val, "aggregate": aggregation}
        for val in values
    ]


def _apply_intervals_to_ocel(
    attribute: Dict[str, Any],
    intervals: List[Tuple[float, float]],
    id_to_cluster: Dict[Any, int] | None
) -> None:
    """
    Saves a new OCEL where the numeric values of one attribute have been
    replaced by their interval labels.

    Inputs:
        attribute  – descriptor dict with keys:
            {
                "qualifier": str,
                "aggregate": str,
                "interval": {
                    "start": float,
                    "end": float
                }
            }
        intervals  – list of (start, end) tuples from discretization

    Output:
        A deep copy of the input OCEL, with the original numeric column
        overwritten by the string "start-end" for values falling in that
        interval.
    """
    ocel = load_ocel(os.path.join(DATA_FOLDER, "ocel.json"))

    attr = attribute.get("attribute", "")
    type = attribute.get("type", "")
    qualifier = attribute.get("qualifier", "")

    if type == "EVENT":
        df = ocel.events
        id_col = "ocel:eid"
        base_mask = df["ocel:activity"] == qualifier
    else:
        df = ocel.objects
        id_col = "ocel:oid"
        base_mask = df["ocel:type"] == qualifier

    df[attr].astype(float)
    temp_col = f"__{attr}__"
    df[temp_col] = None

    interval_labels = [f"[{start}-{end}]" for start, end in intervals]

    if id_to_cluster:
        def map_interval_by_id(eid):
            cid = id_to_cluster.get(eid)
            if cid is None or cid < 0 or cid >= len(interval_labels):
                return None
            return interval_labels[cid]

        df.loc[base_mask, temp_col] = df.loc[base_mask, id_col].map(map_interval_by_id)
    else:
        for idx, (start, end) in enumerate(intervals):
            label = interval_labels[idx]
            mask = base_mask & df[attr].between(start, end, inclusive="both")
            df.loc[mask, temp_col] = label

    save_ocel(ocel, os.path.join(DATA_FOLDER, "ocel.json"))

def transform_ocel(ocel: OCEL, items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a transaction matrix (one-hot encoded) from OCEL events and objects.

    Inputs:
        ocel: OCEL
        items: [{
            "attribute": str,
            "type": str // "EVENT" or "OBJECT",
            "qualifier": str,
            "aggregate": str,
            "value": str // for categorical attributes,
            "interval": {
                "start": float,
                "end": float
            } // for discretized attributes
            "mapping": [str] // eids/oids for this item
        }]

    Output:
        DataFrame where rows are event IDs and columns are item strings; True if present.
    """

    event_groups = {eid: df for eid, df in ocel.events.groupby("ocel:eid")}
    relation_groups = ocel.relations.groupby("ocel:eid")["ocel:oid"].apply(set).to_dict()
    object_groups = {eid: ocel.objects[ocel.objects["ocel:oid"].isin(oids)]for eid, oids in relation_groups.items()}

    transactions: list[list[str]] = []
    event_items = [item for item in items if item.get("type", "") == "EVENT"]
    object_items = [item for item in items if item.get("type", "") == "OBJECT"]

    for eid, ev_df in event_groups.items():
        trans: list[str] = []
        obj_df = object_groups.get(eid, pd.DataFrame())

        for item in event_items:
            if item.get("qualifier", "") not in ev_df["ocel:activity"].values:
                continue

            mapping = item.get("mapping")
            if mapping is not None:
                if eid in mapping:
                    trans.append(f"{item.get('attribute', '')}_{item.get('type', '')}_{item.get('qualifier', '')}_{item.get('interval', {}).get('start', -1)}-{item.get('interval', {}).get('end', -1)}")
                continue

            col = ev_df[item.get("attribute", "")]

            if "interval" in item:
                mask = col.between(item.get("interval", {}).get("start", -1), item.get("interval", {}).get("end", -1))
                if mask.any():
                    trans.append(f"{item.get('attribute', '')}_{item.get('type', '')}_{item.get('qualifier', '')}_{item.get('interval', {}).get('start', -1)}-{item.get('interval', {}).get('end', -1)}")
            elif "value" in item:
                if (col.astype(str) == str(item.get("value", ""))).any():
                    trans.append(f"{item.get('attribute', '')}_{item.get('type', '')}_{item.get('qualifier', '')}_{item.get('value', '')}")

        for item in object_items:
            if item.get("qualifier", '') not in pd.DataFrame(obj_df["ocel:type"]).values:
                continue

            mapping = item.get("mapping")
            if mapping is not None:
                if eid in mapping:
                    trans.append(f"{item.get('attribute', '')}_{item.get('type', '')}_{item.get('qualifier', '')}_{item.get('interval', {}).get('start', -1)}-{item.get('interval', {}).get('end', -1)}")
                continue

            if item.get("aggregate", None):
                col = ev_df[item.get("attribute", "")]
                mask = col.between(item.get("interval", {}).get("start", -1), item.get("interval", {}).get("end", -1))
                if mask.any():
                    trans.append(f"{item.get('attribute', '')}_{item.get('type', '')}_{item.get('qualifier', '')}_{item.get('interval', {}).get('start', -1)}-{item.get('interval', {}).get('end', -1)}")
                continue

            col = pd.Series(obj_df[item.get("attribute", "")])
            if "interval" in item:
                mask = col.between(item.get("interval", {}).get("start", -1), item.get("interval", {}).get("end", -1))
                if mask.any():
                    trans.append(f"{item.get('attribute', '')}_{item.get('type', '')}_{item.get('qualifier', '')}_{item.get('interval', {}).get('start', -1)}-{item.get('interval', {}).get('end', -1)}")
            elif "value" in item:
                if (col.astype(str) == str(item.get("value", ""))).any():
                    trans.append(f"{item.get('attribute', '')}_{item.get('type', '')}_{item.get('qualifier', '')}_{item.get('value', '')}")

        transactions.append(trans)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    col_index = pd.Index(te.columns_, name=None)
    return pd.DataFrame(te_ary, columns=col_index)


def generate_frequent_itemsets(transactions: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """
    Run the Apriori algorithm to find frequent itemsets.

    Inputs:
        transactions: one-hot encoded DataFrame
        min_support: minimum support threshold (0 < min_support <= 1)

    Output:
        DataFrame sorted by descending support with columns ['itemsets', 'support']
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be between 0 (exclusive) and 1 (inclusive)")
    return apriori(transactions, min_support=min_support, use_colnames=True).sort_values("support", ascending=False)



def generate_association_rules(
    frequent_itemsets: pd.DataFrame, min_confidence: float, min_lift: float
) -> pd.DataFrame:
    """
    Derive association rules from frequent itemsets.

    Inputs:
        frequent_itemsets – DataFrame from generate_frequent_itemsets()
        min_confidence    – minimum confidence threshold (0 <= min_confidence <= 1)
        min_lift          – minimum lift threshold (lift >= min_lift)

    Output:
        DataFrame of rules sorted by descending lift
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if min_lift < 0:
        raise ValueError("min_lift must be non-negative")

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    filtered = rules[rules["lift"] >= min_lift]
    return pd.DataFrame(filtered).sort_values("lift", ascending=False)


def frequent_itemset_to_json(frequent_itemsets: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Serialize frequent itemsets to JSON-compatible format.

    Input:
        frequent_itemsets – DataFrame with 'itemsets' and 'support'

    Output:
        [{
            "item": [{
                "attribute": str,
                "type": str // "EVENT" or "OBJECT",
                "qualifier": str,
                "value": str,
            }],
            "support": float
        }]
    """
    result: List[Dict[str, Any]] = []
    for _, row in frequent_itemsets.iterrows():
        support = float(row["support"])
        items = []
        for item_str in row["itemsets"]:
            attr, typ, qual, val = item_str.split("_", 3)
            items.append({
                "attribute": attr,
                "type": typ,
                "qualifier": qual,
                "value": val
            })
        result.append({"item": items, "support": round(support, 4)})
    return result


def association_rule_to_json(association_rules_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Serialize association rules to JSON-compatible format.

    Input:
        association_rules_df – DataFrame with columns:
            "antecedents": frozenset,
            "consequents": frozenset,
            "support",
            "confidence",
            "lift"

    Output:
        [{
            "antecedents": [{
                "attribute": str,
                "type": str // "EVENT" or "OBJECT",
                "qualifier": str,
                "value": str
            }],
            "consequents": [{
                "attribute": str,
                "type": str // "EVENT" or "OBJECT",
                "qualifier": str,
                "value": str
            }],
            "support": float,
            "confidence": float,
            "lift": float
        }]
    """
    result: List[Dict[str, Any]] = []
    for _, row in association_rules_df.iterrows():
        ant_items = []
        for ante in row["antecedents"]:
            attr, typ, qual, val = ante.split("_", 3)
            ant_items.append({
                "attribute": attr,
                "type": typ,
                "qualifier": qual,
                "value": val
            })

        con_items = []
        for cons in row["consequents"]:
            attr, typ, qual, val = cons.split("_", 3)
            con_items.append({
                "attribute": attr,
                "type": typ,
                "qualifier": qual,
                "value": val
            })

        result.append({
            "antecedents": ant_items,
            "consequents": con_items,
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
        })
    return result
