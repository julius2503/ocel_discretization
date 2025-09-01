import copy

from src.mining import run_itemize


def handle_aggregate_attributes(ocel, aggregation_attribute):
    attribute = str(aggregation_attribute["attribute"])
    qualifier = str(aggregation_attribute["qualifier"])
    aggregation = str(aggregation_attribute["aggregation"])

    aggr_relations = ocel.relations[ocel.relations["ocel:type"] == qualifier]
    merged = aggr_relations.merge(ocel.objects[["ocel:oid", attribute]], on="ocel:oid",how="left")
    column_name = f"{aggregation.capitalize()}{attribute.capitalize()}({qualifier.capitalize()})"
    avg = merged.groupby("ocel:eid", as_index=False).agg(**{ column_name: (attribute, aggregation) }).round({column_name: 2})

    ocel = copy.deepcopy(ocel)
    ocel.events = ocel.events.merge(avg, on="ocel:eid", how="left")
    aggregation_attribute["attribute"] = column_name

    return ocel, run_itemize(ocel, aggregation_attribute)
