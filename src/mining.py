from typing import Dict, List

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pm4py import OCEL

from src.discretization import run_discretization
from src.preprocessing import split_numerical_attribute


def run_itemize(ocel:OCEL, attribute):
    attr = attribute["attribute"]
    qualifier = attribute["qualifier"]
    type = attribute["type"]
    splits = attribute.get("split_attributes", None)
    aggregation = attribute["aggregation"]

    match attribute["numeric"]:
        case True:
            match type:
                case "EVENT":
                    events = split_numerical_attribute(ocel, type, splits)

                    values = events[events["ocel:activity"] == qualifier][attr].values.tolist()
                    ids = events[events["ocel:activity"] == qualifier]["ocel:eid"].values.tolist()

                    intervals = run_discretization(ocel, attribute, values, ids)

                case "OBJECT":
                    objects = split_numerical_attribute(ocel, type, splits)

                    if aggregation != "":
                        values = ocel.events[attr].values.tolist()
                    else:
                        values = objects[objects["ocel:type"] == qualifier][attr].values.tolist()

                    ids = objects[objects["ocel:type"] == qualifier]["ocel:oid"].values.tolist()

                    intervals = run_discretization(ocel, attribute, values, ids)

            items = [{
                "attribute": attr,
                "type": type,
                "qualifier": qualifier,
                "interval": {
                    "start": start,
                    "end": end
                },
                "aggregate": aggregation
            } for start, end in intervals]
            return items

        case False:
            match type:
                case "EVENT":
                    values = ocel.events[ocel.events["ocel:activity"] == qualifier][attr].unique()
                case "OBJECT":
                    values = ocel.objects[ocel.objects["ocel:type"] == qualifier][attr].unique()

            items = []
            for value in values:
                items.append({
                    "attribute": attr,
                    "type": type,
                    "qualifier": qualifier,
                    "value": str(value),
                    "aggregate": aggregation
            })
            return items

def transform_ocel(ocel: OCEL, items: List[Dict[str, str]]):
    events = ocel.events
    relations = ocel.relations
    objects = ocel.objects

    event_groups = {eid: df for eid, df in events.groupby("ocel:eid")}
    relation_groups = relations.groupby("ocel:eid")["ocel:oid"].apply(set).to_dict()
    object_groups = {eid: objects[objects["ocel:oid"].isin(oids)]for eid, oids in relation_groups.items()}

    transactions: list[list[str]] = []
    event_items = [item for item in items if item["type"] == "EVENT"]
    object_items = [item for item in items if item["type"] == "OBJECT"]

    for eid, ev_df in event_groups.items():
        trans: list[str] = []
        obj_df = object_groups.get(eid, pd.DataFrame())

        for item in event_items:
            if item["qualifier"] not in ev_df["ocel:activity"].values:
                continue

            col = ev_df[item["attribute"]]
            if "interval" in item:
                mask = col.between(item["interval"]["start"], item["interval"]["end"])
                if mask.any():
                    trans.append(f"{item['attribute']}_{item['type']}_{item['qualifier']}_{item['interval']['start']}-{item['interval']['end']}")
            elif "value" in item:
                if (col.astype(str) == str(item["value"])).any():
                    trans.append(f"{item['attribute']}_{item['type']}_{item['qualifier']}_{item['value']}")

        for item in object_items:
            if item["qualifier"] not in obj_df["ocel:type"].values:
                continue

            if item["aggregate"]:
                col = ev_df[item["attribute"]]
                mask = col.between(item["interval"]["start"], item["interval"]["end"])
                if mask.any():
                    trans.append(f"{item['attribute']}_{item['type']}_{item['qualifier']}_{item['interval']['start']}-{item['interval']['end']}")
                continue

            col = obj_df[item["attribute"]]
            if "interval" in item:
                mask = col.between(item["interval"]["start"], item["interval"]["end"])
                if mask.any():
                    trans.append(f"{item['attribute']}_{item['type']}_{item['qualifier']}_{item['interval']['start']}-{item['interval']['end']}")
            elif "value" in item:
                if (col == item["value"]).any():
                    trans.append(f"{item['attribute']}_{item['type']}_{item['qualifier']}_{item['value']}")

        transactions.append(trans)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def generate_frequent_itemsets(transactions, min_support):
    return apriori(
        transactions, min_support=min_support, use_colnames=True
    ).sort_values("support", ascending=False)


def generate_association_rules(frequent_itemsets, min_confidence, min_lift):
    rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

    rules = rules[rules["lift"] >= min_lift]

    return rules.sort_values("lift", ascending=False)


def frequent_itemset_to_json(frequent_itemsets):
    result = []
    for _, row in frequent_itemsets.iterrows():
        support = float(row["support"])
        items = list(row["itemsets"])
        item_objects = []

        for item_str in items:
            attribute, type, qualifier, value = item_str.split("_")
            item_objects.append(
                {
                    "attribute": attribute,
                    "type": type,
                    "qualifier": qualifier,
                    "value": value,
                }
            )

        result.append({"item": item_objects, "support": round(support, 4)})

    return result


def association_rule_to_json(association_rules):
    result = []

    for _, row in association_rules.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])

        ant = []
        for antecedent in antecedents:
            attribute, type, qualifier, value = antecedent.split("_")
            ant.append(
                {
                    "attribute": attribute,
                    "type": type,
                    "qualifier": qualifier,
                    "value": value,
                }
            )

        con = []
        for consequent in consequents:
            attribute, type, qualifier, value = consequent.split("_")
            con.append(
                {
                    "attribute": attribute,
                    "type": type,
                    "qualifier": qualifier,
                    "value": value,
                }
            )

        result.append(
            {
                "antecedents": ant,
                "consequents": con,
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
            }
        )

    return result
