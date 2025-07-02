import pandas as pd
import numpy as np
import pm4py
from pm4py import OCEL
from typing import List
import copy
import chimerge
from scipy.stats import chi2
from sklearn.cluster import KMeans

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
        
def get_attribute_values(ocel: OCEL, attribute, qualifier, type):
    match type:
        case "EVENT":
            values = ocel.events[ocel.events["ocel:activity"] == qualifier][attribute].unique().tolist()
            return [{"value": str(v)} for v in values]
        
        case "OBJECT":
            values = ocel.objects[ocel.objects["ocel:type"] == qualifier][attribute].unique().tolist()
            return [{"value": str(v)} for v in values]

        
def get_attributes(ocel: OCEL) -> List[List[str]]:
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

def not_numeric(ocel:OCEL, data):
    attribute = data["attribute"]
    type = data["type"]
    qualifier = data["qualifier"]

    match type:
        case "EVENT":
            values = ocel.events[ocel.events["ocel:activity"] == qualifier][attribute].unique()
        case "OBJECT":
            values = ocel.objects[ocel.objects["ocel:type"] == qualifier][attribute].unique()

    items = []
    for value in values:
        items.append([{
            "attribute": attribute, 
            "type": type,
            "qualifier": qualifier,
            "value": str(value)
    }])
    return items

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
        
def handle_aggregate_attributes(ocel, related_attribute, algorithm, parameters):
    aggr_attribute = related_attribute["attribute"]
    aggr_qualifier = related_attribute["qualifier"]
    aggr_function = related_attribute["aggregate"]

    aggr_relations = ocel.relations[ocel.relations["ocel:type"] == aggr_qualifier]
    merged = aggr_relations.merge(ocel.objects[["ocel:oid", aggr_attribute]], on="ocel:oid",how="left")
    avg = merged.groupby("ocel:eid", as_index=False).agg(Aggr=(aggr_attribute, aggr_function)).round({"Aggr": 2})
                        
    aggr_ocel = copy.deepcopy(ocel)
    aggr_ocel.events = aggr_ocel.events.merge(avg, on="ocel:eid", how="left")

    return run_discretization(ocel,aggr_ocel.events["Aggr"].dropna().values.tolist(), aggr_ocel.events, algorithm, parameters)



def run_itemize(ocel:OCEL, data, algorithm):
    attribute = data["attribute"]
    type = data["type"]
    qualifier = data["qualifier"]
    related = [related for related in data["related"] if related["selected"]]
    splits = data["split_attributes"]
    algorithm_name = algorithm["name"]
    algorithm_parameters = algorithm["parameters"]

    match type:
        case "EVENT":
            events = split_numerical_attribute(ocel, type, splits)

            values = events[events["ocel:activity"] == qualifier][attribute].values.tolist()

            intervals = run_discretization(ocel, values, events[events["ocel:activity"] == qualifier], algorithm_name, algorithm_parameters)

        case "OBJECT":
            objects = split_numerical_attribute(ocel, type, splits)

            values = objects[objects["ocel:type"] == qualifier][attribute].values.tolist()

            intervals = run_discretization(ocel, values, objects[objects["ocel:type"] == qualifier] ,algorithm_name, algorithm_parameters)

    items = [[{ 
        "attribute": attribute, 
        "type": type,
        "qualifier": qualifier,
        "interval": {
            "start": start,
            "end": end
        }
    }] for start, end in intervals]

    for related_attribute in related:
        events = ocel.events
        objects = ocel.objects
        match type:
            case "EVENT":
                match related_attribute["type"]:
                    case "EVENT":
                        values = events[events["ocel:activity"] == related_attribute["qualifier"]][related_attribute["attribute"]].unique()
                        new_items = []
                        for value in values:
                            for item in items:
                                new_items.append(item + [{
                                    "attribute": related_attribute["attribute"], 
                                    "type": "EVENT",
                                    "qualifier": qualifier,
                                    "value": str(value)
                                    }])
                        items = new_items                   
                    
                    case "OBJECT":
                        values = objects[objects["ocel:type"] == related_attribute["qualifier"]][related_attribute["attribute"]].unique().tolist()
                        if related_attribute["aggregate"]:
                            values = handle_aggregate_attributes(ocel, related_attribute, algorithm_name, algorithm_parameters)
                        new_items = []
                        for value in values:
                            for item in items:
                                new_items.append(item + [{
                                    "attribute": related_attribute["attribute"], 
                                    "type": "OBJECT",
                                    "qualifier": qualifier,
                                    "aggregate": related_attribute["aggregate"],
                                    "value": str(value)
                                    }])
                        items = new_items 
            
            case "OBJECT":
                values = objects[objects["ocel:type"] == related_attribute["qualifier"]][related_attribute["attribute"]].unique()
                new_items = []
                for value in values:
                    for item in items:
                        new_items.append(item + [{
                            "attribute": related_attribute["attribute"], 
                            "type": "OBJECT",
                            "qualifier": qualifier,
                            "value": str(value)
                            }])
                items = new_items
    return items

def run_discretization(ocel, values, events, algorithm_name, algorithm_parameters):
    match algorithm_name:
        case "equal-freq":
            bins = int(algorithm_parameters["bins"])
            return equal_frequency_binning(values, bins)
        case "equal-width":
            bins = int(algorithm_parameters["bins"])
            return equal_width_binning(values, bins)
        case "chi-merge":
            labels = algorithm_parameters["labels"]
            interval = int(algorithm_parameters["max_interval"])
            significance = float(algorithm_parameters["significance"])
            return chi_merge_binning(values, events, labels, interval, significance)
        case "k-means":
            cluster = int(algorithm_parameters["clusters"])
            return kmeans_clustering(values, cluster)

def equal_frequency_binning(values, bins):
    partitions = np.array_split(sorted(values), int(bins))
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

    return intervals

def equal_width_binning(values, n_bins):
    values = np.array(values)
    min_val = float(values.min())
    max_val = float(values.max())
    bin_width = (max_val - min_val) / n_bins
    bins = [min_val + i * bin_width for i in range(n_bins)]
    
    intervals = []
    for i in range(n_bins):
        start = bins[i]
        end = bins[i+1] if i < n_bins-1 else max_val
        intervals.append((int(start), int(end)))
    
    return intervals


def chi_merge_binning(values, events, labels, n_intervals=6, alpha=0.1, min_gap=1.0):
    labels = events[labels[0]].astype(str).tolist()

    intervals = chimerge.initialize_intervals(values, labels)
    all_labels = sorted(list(set(labels)))
    df = len(all_labels) - 1
    threshold = chi2.ppf(1 - alpha, df) if df > 0 else 0

    while len(intervals) > n_intervals:
        min_chi2, min_idx = float('inf'), -1
        for i in range(len(intervals) - 1):
            chi2_val = chimerge.compute_chi2(intervals[i], intervals[i+1], all_labels)
            if chi2_val < min_chi2:
                min_chi2, min_idx = chi2_val, i
        if min_chi2 > threshold:
            break
        merged = chimerge.merge_intervals(intervals[min_idx], intervals[min_idx+1])
        intervals = intervals[:min_idx] + [merged] + intervals[min_idx+2:]
    
    while len(intervals) > n_intervals:
        min_chi2, min_idx = float('inf'), -1
        for i in range(len(intervals) - 1):
            chi2_val = chimerge.compute_chi2(intervals[i], intervals[i+1], all_labels)
            if chi2_val < min_chi2:
                min_chi2, min_idx = chi2_val, i
        merged = chimerge.merge_intervals(intervals[min_idx], intervals[min_idx+1])
        intervals = intervals[:min_idx] + [merged] + intervals[min_idx+2:]
    
    cut_points = sorted(set(interval['end'] for interval in intervals[:-1]))
    cut_points = [float(cp) for cp in cut_points]
    min_val, max_val = min(values), max(values)
    intervals = []
    if cut_points:
        intervals.append((float(min_val), float(cut_points[0])))
        for i in range(len(cut_points)-1):
            intervals.append((float(cut_points[i]), float(cut_points[i+1])))
        intervals.append((float(cut_points[-1]), float(max_val)))
    else:
        intervals.append((float(min_val), float(max_val)))
        
    return intervals

def kmeans_clustering(values, n_clusters, random_state=42):
    X = np.array(values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    
    centers = np.sort(kmeans.cluster_centers_.flatten())
    min_val = float(min(values))
    max_val = float(max(values))
    
    intervals = []
    if len(centers) > 1:
        intervals.append((round(min_val, 2), round(float(centers[0]), 2)))
        for i in range(len(centers)-1):
            intervals.append((round(float(centers[i]), 2), round(float(centers[i+1]), 2)))
        intervals.append((round(float(centers[-1]), 2), round(max_val, 2)))
    else:
        intervals.append((round(min_val, 2), round(max_val, 2)))
    
    return intervals
