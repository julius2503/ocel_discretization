import os
from typing import Any, Dict, List

import pandas as pd
import pm4py
from pm4py import OCEL


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    """
    Check if the filename has an allowed extension.

    Inputs:
        filename           – name of the file (e.g. 'log.json')
        allowed_extensions – set or list of extensions (e.g. {'json','sqlite'})

    Output:
        True if extension is allowed, False otherwise.
    """
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[-1].lower() in allowed_extensions


def load_ocel(file_path: str) -> OCEL:
    """
    Load an OCEL from JSON or SQLite file.

    Input:
        file_path – path to .json or .sqlite OCEL file

    Output:
        OCEL instance

    Raises:
        FileNotFoundError if file does not exist.
        ValueError if extension is unsupported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"OCEL file not found: {file_path}")

    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "json":
        return pm4py.read_ocel2_json(file_path=file_path)
    if ext == "sqlite":
        return pm4py.read_ocel2_sqlite(file_path=file_path)
    if ext == "xml":
        return pm4py.read_ocel2_xml(file_path=file_path)
    raise ValueError(f"Unsupported OCEL file type: .{ext}")


def save_ocel(ocel: OCEL, file_path: str) -> None:
    """
    Serialize an OCEL instance to JSON format.

    Inputs:
        ocel      – OCEL instance
        file_path – destination path ending in .json

    Raises:
        IOError on write failures.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    pm4py.write_ocel2_json(ocel, file_path)


def _populate_related(ocel: OCEL, descriptors: List[Dict[str, Any]]) -> None:
    """Fill each descriptor's `related` list with same- and cross-entity attrs."""
    events = pd.DataFrame(ocel.events)
    objects = pd.DataFrame(ocel.objects)

    def _vals(attr: str, qualifier: str, type: str) -> List[Dict[str, str]]:
        df = events if type == "EVENT" else objects
        col = "ocel:activity" if type == "EVENT" else "ocel:type"
        series = pd.DataFrame(df[df[col] == qualifier])[attr].dropna().unique()
        return [{"value": str(v)} for v in series]

    for base in descriptors:
        attr = base.get("attribute", "")
        type = base.get("type", "")
        qualifier = base.get("qualifier", "")

        for other in descriptors:
            if other.get("type") == type and other.get("qualifier") == qualifier and other.get("attribute") != attr:
                base.get("related", []).append(
                    {
                        "attribute": other.get("attribute", ""),
                        "type": other.get("type", ""),
                        "qualifier": other.get("qualifier", ""),
                        "vals": _vals(other.get("attribute", ""), qualifier, type),
                    }
                )

        rel = ocel.relations
        if type == "EVENT":
            for obj_t in pd.DataFrame(rel[rel["ocel:activity"] == qualifier])["ocel:type"].unique():
                for other in descriptors:
                    if other.get("type") == "OBJECT" and other.get("qualifier") == obj_t:
                        base.get("related", []).append(
                            {
                                "attribute": other.get("attribute", ""),
                                "type": "OBJECT",
                                "qualifier": obj_t,
                                "vals": _vals(other.get("attribute", ""), obj_t, "OBJECT"),
                            }
                        )
        else:
            for act in pd.DataFrame(rel[rel["ocel:type"] == qualifier])["ocel:activity"].unique():
                for other in descriptors:
                    if other.get("type") == "EVENT" and other.get("qualifier") == act:
                        base.get("related", []).append(
                            {
                                "attribute": other.get("attribute", ""),
                                "type": "EVENT",
                                "qualifier": act,
                                "vals": _vals(other.get("attribute", ""), act, "EVENT"),
                            }
                        )


def get_attributes(ocel: OCEL) -> List[Dict[str, Any]]:
    """
    Extract all event and object attributes, with their qualifiers and related attrs.

    Input:
      ocel – OCEL instance

    Output:
        [{
            "attribute": str,
            "type": str // "EVENT" or "OBJECT",
            "qualifier": str,
            "related": [{
                    "attribute": str,
                    "type": str // "EVENT" or "OBJECT",
                    "qualifier": str,
                    "vals": [{
                        "value": str
                    }]
            }]
        }]
    """
    events = pd.DataFrame(ocel.events)
    objects = pd.DataFrame(ocel.objects)

    event_core = {"ocel:eid", "ocel:timestamp", "ocel:activity"}
    object_core = {"ocel:oid", "ocel:timestamp", "ocel:type"}

    descriptors: List[Dict[str, Any]] = []

    for attr in [c for c in events.columns if c not in event_core]:
        quals = pd.DataFrame(events[events[attr].notna()])["ocel:activity"].unique()
        for qual in quals:
            descriptors.append({"attribute": attr, "type": "EVENT", "qualifier": qual, "related": []})

    for attr in [c for c in objects.columns if c not in object_core]:
        quals = pd.DataFrame(objects[objects[attr].notna()])["ocel:type"].unique()
        for qual in quals:
            descriptors.append({"attribute": attr, "type": "OBJECT", "qualifier": qual, "related": []})

    _populate_related(ocel, descriptors)
    return descriptors


def get_attribute_values(ocel: OCEL, attribute: str, qualifier: str, entity_type: str) -> List[Dict[str, str]]:
    """
    Get distinct values of an attribute for a given event activity or object type.

    Inputs:
      ocel         – OCEL instance
      attribute    – column name
      qualifier    – activity name if EVENT or object type if OBJECT
      entity_type  – "EVENT" or "OBJECT"

    Output:
      List of {"value": str} for each unique non-null value.
    """
    if entity_type == "EVENT":
        vals = ocel.events.loc[ocel.events["ocel:activity"] == qualifier, attribute].dropna().unique()
    elif entity_type == "OBJECT":
        vals = ocel.objects.loc[ocel.objects["ocel:type"] == qualifier, attribute].dropna().unique()
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    return [{"value": str(v)} for v in vals]


def split_numerical_attribute(ocel: OCEL, type: str, splits: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Filter events or objects by split criteria before numeric discretization.

    Inputs:
      ocel         – OCEL instance
      type  – "EVENT" or "OBJECT"
      splits       – list of dicts:
        {"type":"EVENT"/"OBJECT","attribute":str,"qualifier":str,"selected_value":str}

    Output:
      Filtered DataFrame of events or objects for numeric discretization.
    """
    if type == "EVENT":
        df = ocel.events.copy()
        rel = ocel.relations
        for split in splits:
            val = split.get("selected_value", "")
            if val is None:
                continue
            if split.get("type", "") == "EVENT":
                if not split.get("agg"):
                    df = pd.DataFrame(df[df[split.get("attribute", "")].astype(str) == val])
                else:
                    start, end = val.split(", ")
                    start, end = float(start.strip("[")), float(end.strip("]"))
                    df = pd.DataFrame(df[df[split.get("attribute", "")].astype(float).between(start, end)])
            else:
                oids = ocel.objects[
                    (ocel.objects["ocel:type"] == split.get("qualifier", "")) & (ocel.objects[split.get("attribute", "")].astype(str) == val)
                ]["ocel:oid"]
                eids = rel[rel["ocel:oid"].isin(pd.Series(oids))]["ocel:eid"]
                df = pd.DataFrame(df[pd.Series(df["ocel:eid"]).isin(pd.Series(eids))])
        return df

    if type == "OBJECT":
        df = ocel.objects.copy()
        for split in splits:
            val = split.get("selected_value", "")
            if val is None:
                continue
            df = pd.DataFrame(df[(df["ocel:type"] == split.get("qualifier", "")) & (df[split.get("attribute", "")].astype(str) == val)])
        return df

    raise ValueError(f"Unknown entity_type: {type}")
