import logging
import os
from typing import List

import pandas as pd
from pm4py import OCEL

from config import DATA_FOLDER
from src.mining import run_itemize
from src.preprocessing import save_ocel

logger = logging.getLogger(__name__)

def handle_aggregate_attributes(ocel: OCEL, aggregation_attribute: dict) -> List[dict]:
    """
    Aggregate an object attribute over related events.

    Input:
        ocel: OCEL instance
        aggregation_attribute: {
        "attribute": str,       # object attribute name
        "qualifier": str,       # object type
        "aggregation": str      # aggregation function ("mean", "sum", etc.)
        }

    Output:
        Tuple[updated_ocel, items]
        - updated_ocel: deep-copied OCEL with new aggregated column on events
        - items: item definitions for subsequent mining
    """
    attribute = aggregation_attribute.get("attribute", "")
    qualifier = aggregation_attribute.get("qualifier", "")
    agg_func = aggregation_attribute.get("aggregation", "")

    if agg_func not in {"mean", "median", "sum", "min", "max", "count"}:
        raise ValueError(f"Unsupported aggregation: {agg_func}")

    rels = ocel.relations[ocel.relations["ocel:type"] == qualifier]
    merged = rels.merge(ocel.objects[[ocel.object_id_column, attribute]], on=ocel.object_id_column, how="left")

    column_name = f"{agg_func.capitalize()}{attribute.capitalize()}({qualifier.capitalize()})"
    grouped = pd.DataFrame(
        merged.groupby("ocel:eid", as_index=False)
                .agg(**{column_name: (attribute, agg_func)})
    )
    grouped[column_name] = grouped[column_name].round(2)

    ocel.events = ocel.events.merge(grouped, on="ocel:eid", how="left")

    aggregation_attribute.update(attribute=column_name)

    items = run_itemize(ocel, aggregation_attribute)

    save_ocel(ocel, os.path.join(DATA_FOLDER, "ocel.json"))
    return items
