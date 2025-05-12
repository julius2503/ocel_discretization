import pandas as pd
import numpy as np
import pm4py
from pm4py import OCEL
from typing import List, Dict, Any


def load_ocel(file_path: str) -> OCEL:
    file_type = file_path.split(".")[1]
    
    match file_type:
        case "json":
            return pm4py.read_ocel2_json(file_path=file_path)
        case "sqlite":
            return pm4py.read_ocel2_sqlite(file_path=file_path)
        case _: 
            return pm4py.read_ocel2(file_path=file_path)

def get_attributes(ocel: OCEL) -> List[List[str]]:
    events = ocel.events
    objects = ocel.objects

    event_cols = {'ocel:eid', 'ocel:timestamp', 'ocel:activity'}
    object_cols = {'ocel:oid', 'ocel:timestamp', 'ocel:type'}

    attributes = [col for col in events.columns if col not in event_cols]
    
    result = []
    
    for attr in attributes:
        non_na_activities = events[events[attr].notna()]['ocel:activity'].unique()
        
        for activity in non_na_activities:
            result.append([activity, "EVENT", attr])

    attributes = [col for col in objects.columns if col not in object_cols]
    
    for attr in attributes:
        non_na_activities = objects[objects[attr].notna()]['ocel:type'].unique()
        
        for activity in non_na_activities:
            result.append([activity, "OBJECT" ,attr])

    return sorted(result)

def get_related_attributes(ocel:OCEL, attribute:str) -> List[str]:
    name, type, attr = attribute.split(",")

    attributes = get_attributes(ocel=ocel)
    related_object_types = []

    if type == "EVENT":
        for value in [value[2] for value in attributes if value[0] == name]:
            if value != attr:
                related_object_types.append(f"{name}, 'EVENT', {value}")

        e2o = ocel.relations[ocel.relations["ocel:activity"] == name]
        related_objects = e2o["ocel:type"].unique()

        for object in related_objects:
            for value in [value[2] for value in attributes if value[0] == object]:
                related_object_types.append(f"{object}, 'OBJECT', {value}")

    elif type == "OBJECT":
        for value in [value[2] for value in attributes if value[0] == name]:
            if value != attr:
                related_object_types.append(f"{name}, 'OBJECT', {value}")

        o2o = o2o_mapping(ocel=ocel)
        o2o = o2o[o2o["source"] == name]
        related_objects = o2o["target"].unique()
        for object in related_objects:
            for value in [value[2] for value in attributes if value[0] == object]:
                related_object_types.append(f"{object}, 'OBJECT', {value}")
    
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

def run_equal_frequency_binning(ocel:OCEL, attribute:str, params: int | str, related_attr:List[str]) -> Dict[str, Any]:
    name, type, attr = attribute.split(",")
    
    match type:
        case "EVENT":
            events = ocel.events[ocel.events["ocel:activity"] == name]
            values = sorted(events[attr].values.tolist())
        case "OBJECT":
            objects = ocel.objects[ocel.objects["ocel:type"] == name]
            values = sorted(objects[attr].values.tolist())
        case _:
            raise Exception(f"{type} ist weder Event noch Object!")

    partitions = np.array_split(values, int(params))
    intervals = []
    prev_end = None
    
    for part in partitions:
        if len(part) == 0:
            continue
            
        current_start = part[0].item()
        current_end = part[-1].item()
        
        if prev_end is not None:
            current_start = prev_end
            
        intervals.append((current_start, current_end))
        prev_end = current_end

    interval_objects = [
        pd.Interval(left=start, right=end, closed='right') 
        for start, end in intervals
    ]

    return {
        f"{name}-{attr}":{
            "type": type,
            "count": len(values),
            "intervals": interval_objects
        }
    }

def run_equal_width_binning(ocel, attribute, params):
    pass

def run_chi_merge_binning(ocel, attribute, params):
    pass

def run_kmeans_clustering(ocel, attribute, params):
    pass