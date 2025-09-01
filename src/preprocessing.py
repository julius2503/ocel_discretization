from typing import List

import pandas as pd
import pm4py
from pm4py import OCEL


def allowed_file(filename: str, allowed_extensions: str) -> bool:
    return '.' in filename and filename.split(".")[1] in allowed_extensions

def load_ocel(file_path: str) -> OCEL:
    file_type = file_path.split(".")[1]

    match file_type:
        case "json":
            return pm4py.read_ocel2_json(file_path=file_path)
        case "sqlite":
            return pm4py.read_ocel2_sqlite(file_path=file_path)
        case _:
            return pm4py.read_ocel2(file_path=file_path)

def save_ocel(ocel: OCEL, file_path: str) -> None:
    pm4py.write_ocel2_json(ocel, file_path)

def get_attribute_values(ocel: OCEL, attribute, qualifier, type):
    match type:
        case "EVENT":
            values = ocel.events[ocel.events["ocel:activity"] == qualifier][attribute].unique().tolist()
            return [{"value": str(v)} for v in values]

        case "OBJECT":
            values = ocel.objects[ocel.objects["ocel:type"] == qualifier][attribute].unique().tolist()
            return [{"value": str(v)} for v in values]


def get_attributes(ocel: OCEL) -> List[List[str]]:
    events = pd.DataFrame(ocel.events)
    objects = pd.DataFrame(ocel.objects)

    event_cols = {'ocel:eid', 'ocel:timestamp', 'ocel:activity'}
    object_cols = {'ocel:oid', 'ocel:timestamp', 'ocel:type'}

    attributes = [col for col in events.columns if col not in event_cols]

    result = []

    for attr in attributes:
        attribute = attr
        qualifier = events[events[attr].notna()]['ocel:activity'].unique()
        type = "EVENT"

        for activity in qualifier:
            result.append(
                {
                    "attribute": attribute,
                    "type": type,
                    "qualifier": activity,
                    "related" : get_related_attributes(ocel, attribute, type, activity)
                }
            )

    attributes = [col for col in objects.columns if col not in object_cols]

    for attr in attributes:
        attribute = attr
        qualifier = objects[objects[attr].notna()]['ocel:type'].unique()
        type = "OBJECT"

        for activity in qualifier:
            result.append(
                {
                    "attribute": attribute,
                    "type": type,
                    "qualifier": activity,
                    "related" : get_related_attributes(ocel, attribute, type, activity)
                }
            )

    return result

def get_attribute_list(ocel: OCEL):
    events = ocel.events
    objects = ocel.objects

    event_cols = {'ocel:eid', 'ocel:timestamp', 'ocel:activity'}
    object_cols = {'ocel:oid', 'ocel:timestamp', 'ocel:type'}

    attributes = [col for col in events.columns if col not in event_cols]

    result = []

    for attr in attributes:
        attribute = attr
        qualifier = events[events[attr].notna()]['ocel:activity'].unique()
        type = "EVENT"

        for activity in qualifier:
            result.append(
                {
                    "attribute": attribute,
                    "type": type,
                    "qualifier": activity,
                }
            )

    attributes = [col for col in objects.columns if col not in object_cols]

    for attr in attributes:
        attribute = attr
        qualifier = objects[objects[attr].notna()]['ocel:type'].unique()
        type = "OBJECT"

        for activity in qualifier:
            result.append(
                {
                    "attribute": attribute,
                    "type": type,
                    "qualifier": activity
                }
            )

    return result

def get_related_attributes(ocel:OCEL, attribute:str, type: str, qualifier: str) -> List[str]:

    attributes = get_attribute_list(ocel=ocel)

    related_object_types = []

    if type == "EVENT":
        for value in [value["attribute"] for value in attributes if value["qualifier"] == qualifier]:
            if value != attribute:
                related_object_types.append(
                    {
                        "attribute": value,
                        "type": "EVENT",
                        "qualifier": qualifier,
                        "vals": get_attribute_values(ocel, value, qualifier, "EVENT")
                    }
                )

        e2o = ocel.relations[ocel.relations["ocel:activity"] == qualifier]
        related_objects = e2o["ocel:type"].unique()

        for object in related_objects:
            for value in [value["attribute"] for value in attributes if value["qualifier"] == object]:
                related_object_types.append(
                    {
                        "attribute": value,
                        "type": "OBJECT",
                        "qualifier": object,
                        "vals": get_attribute_values(ocel, value, object, "OBJECT")
                    }
                )

    elif type == "OBJECT":
        for value in [value["attribute"] for value in attributes if value["qualifier"] == qualifier]:
            if value != attribute:
                related_object_types.append(
                    {
                        "attribute": value,
                        "type": type,
                        "qualifier": qualifier,
                        "vals": get_attribute_values(ocel, value, qualifier, type)
                    }
                )

        o2o = o2o_mapping(ocel=ocel)
        o2o = o2o[o2o["source"] == qualifier]
        related_objects = o2o["target"].unique()
        for object in related_objects:
            for value in [value["attribute"] for value in attributes if value["qualifier"] == object]:
                related_object_types.append(
                    {
                        "attribute": value,
                        "type": type,
                        "qualifier": object,
                        "vals": get_attribute_values(ocel, value, object, type)
                    }
                )

    return related_object_types

def o2o_mapping(ocel: OCEL) -> pd.DataFrame:
    object_types = ocel.objects[[ocel.object_id_column, ocel.object_type_column]]
    oid_to_otype = dict(zip(
        object_types[ocel.object_id_column],
        object_types[ocel.object_type_column]
    ))
    o2o_relations = ocel.o2o.copy()

    o2o_relations["source"] = o2o_relations[ocel.object_id_column].map(oid_to_otype)
    o2o_relations["target"] = o2o_relations[ocel.object_id_column + "_2"].map(oid_to_otype)

    return o2o_relations

def split_numerical_attribute(ocel: OCEL, type, splits):
    match type:
        case "EVENT":
            events = ocel.events
            objects = ocel.objects
            relations = ocel.relations

            for split in splits:
                if split["selected_value"] is None:
                    continue
                match split["type"]:
                    case "EVENT":
                        events = events[events[split["attribute"]].astype(str) == split["selected_value"]]

                    case "OBJECT":
                        fitting_oids = objects[(objects["ocel:type"] == split["qualifier"]) & (objects[split["attribute"]].astype(str) == split["selected_value"])]["ocel:oid"].tolist()
                        fitting_eids = relations[relations["ocel:oid"].isin(fitting_oids)]["ocel:eid"].unique().tolist()
                        events = events[events["ocel:eid"].isin(fitting_eids)]

                    case _:
                        raise Exception("Weder EVENT noch OBJECTS")

            return events

        case "OBJECT":
            objects = ocel.objects

            for split in splits:
                objects = objects[(objects["ocel:type"] == split["qualifier"]) & (objects[split["attribute"]].astype(str) == split["selected_value"])]

            return objects

        case _:
            raise Exception("Weder EVENT noch OBJECTS")
