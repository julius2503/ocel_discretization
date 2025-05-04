import os
import pandas as pd
import numpy as np
from flask import Flask, flash, request, redirect, render_template, session
from werkzeug.utils import secure_filename
import pm4py
from pm4py import OCEL
from typing import List, Dict, Any

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json', 'sqlite'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'key'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.split(".")[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'] == '':
            flash('Keine Datei ausgewählt')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            session['current_ocel'] = file_path

            try:
                ocel = load_ocel(file_path)
                attributes = get_attributes(ocel)
                
                return render_template('results.html', 
                                      filename=filename,
                                      attributes=attributes)
            except Exception as e:
                flash(f'Fehler: {str(e)}')
                return redirect(request.url)

    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_attributes():
    algorithm = request.form.get('algorithm')
    selected_attrs = request.form.getlist('selected_attributes')

    file_path = session.get('current_ocel')
    if not file_path:
        flash('Fehler beim Laden des OCEL')
        return redirect(request.url)
    ocel = load_ocel(file_path)

    results = {
        "attributes": []
    }

    match algorithm:
        case "equal_freq":
            for i, attribute in enumerate(selected_attrs):
                params = request.form.get(f"params[{i}][bins]")
                results["attributes"].append(run_equal_frequency_binning(ocel, attribute, params)) # type:ignore
        case "equal_width":
            pass
        case "chi_merge":
            pass
        case "kmeans":
            pass
        case _:
            flash(f'Algorithmus {algorithm} konnte nicht gefunden werden!')
            return redirect(request.url)

    return render_template('items.html', results=results)

def get_attributes(ocel: OCEL) -> List[List[str]]:
    events = ocel.events
    objects = ocel.objects

    event_cols = {'ocel:eid', 'ocel:timestamp', 'ocel:activity'}
    object_cols = {'ocel:oid', 'ocel:timestamp', 'ocel:type'}

    attributes = [col for col in events.columns if col not in event_cols]
    
    result = []
    
    for attr in attributes:
        is_numeric = pd.api.types.is_numeric_dtype(events[attr])
        numeric_str = 'yes' if is_numeric else 'no'
        non_na_activities = events[events[attr].notna()]['ocel:activity'].unique()
        
        for activity in non_na_activities:
            result.append([activity, "EVENT", attr, numeric_str])

    attributes = [col for col in objects.columns if col not in object_cols]
    
    for attr in attributes:
        is_numeric = pd.api.types.is_numeric_dtype(objects[attr])
        numeric_str = 'yes' if is_numeric else 'no'
        non_na_activities = objects[objects[attr].notna()]['ocel:type'].unique()
        
        for activity in non_na_activities:
            result.append([activity, "OBJECT" ,attr, numeric_str])

    return sorted(result)

def load_ocel(file_path: str) -> OCEL:
    file_type = file_path.split(".")[1]
    
    match file_type:
        case "json":
            return pm4py.read_ocel2_json(file_path=file_path)
        case "sqlite":
            return pm4py.read_ocel2_sqlite(file_path=file_path)
        case _: 
            raise Exception(f"Filetype {file_type} wird nicht unterstützt!")

        
def run_equal_frequency_binning(ocel: OCEL, attribute:str, params: int | str) -> Dict[str, Any]:
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

if __name__ == '__main__':
    app.run(debug=True)
