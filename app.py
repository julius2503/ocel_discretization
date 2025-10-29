import json
import logging
import os

import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, DATA_FOLDER, MAX_CONTENT_LENGTH, UPLOAD_FOLDER
from src.aggregation import handle_aggregate_attributes
from src.mining import (
    _apply_intervals_to_ocel,
    association_rule_to_json,
    frequent_itemset_to_json,
    generate_association_rules,
    generate_frequent_itemsets,
    run_itemize,
    transform_ocel,
)
from src.preprocessing import allowed_file, get_attributes, load_ocel, save_ocel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev_key"),
)

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.template_filter("file_extension")
def file_extension(filename):
    """Extract uppercase file extension or return empty string."""
    return filename.rsplit(".", 1)[-1].upper() if "." in filename else ""


@app.route("/", methods=["GET", "POST"])
def upload_file():
    """
    Display upload form and handle file uploads or selection of existing files.
    On success, loads OCEL, saves to data/ocel.json, and presents attribute selection.
    """
    existing_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.rsplit(".", 1)[-1] in ALLOWED_EXTENSIONS]

    if request.method == "POST":
        # Handle new upload
        if "file" in request.files:
            file = request.files["file"]
            if not file or file.filename == "":
                flash("No file selected")
                return redirect(request.url)
            if file.filename and allowed_file(file.filename, ALLOWED_EXTENSIONS):
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)
            else:
                flash("Unsupported file type")
                return redirect(request.url)
        # Handle existing file selection
        elif "existing_file" in request.form:
            filename = request.form["existing_file"]
            path = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.isfile(path):
                flash("Selected file no longer exists")
                return redirect(request.url)
        else:
            flash("No action specified")
            return redirect(request.url)

        try:
            ocel = load_ocel(path)
            save_ocel(ocel, os.path.join(DATA_FOLDER, "ocel.json"))
            attributes = get_attributes(ocel)
            return render_template("select.html", filename=filename, attributes=attributes)
        except Exception as e:
            logger.exception("Failed to load OCEL")
            flash(f"Error loading file: {e}")
            return redirect(request.url)

    return render_template("index.html", existing_files=existing_files)


@app.route("/process", methods=["POST"])
def process_attributes():
    """
    Receives selected and aggregated attributes, applies itemization
    and saves items.json. Returns JSON redirect to show_items.
    """
    data = request.get_json(force=True)
    attrs = data.get("attributes", [])

    with open(os.path.join(DATA_FOLDER, "attributes.json"), "w") as f:
        json.dump(attrs, f)

    items = []

    ocel = load_ocel(os.path.join(DATA_FOLDER, "ocel.json"))

    for attr in attrs:
        if attr.get("aggregation"):
            ocel = load_ocel(os.path.join(DATA_FOLDER, "ocel.json"))
            ocel, new_items = handle_aggregate_attributes(ocel, attr)
            items.extend(new_items)
            save_ocel(ocel, os.path.join(DATA_FOLDER, "ocel.json"))

    for attr in attrs:
        if attr.get("selected") and not attr.get("aggregation"):
            items.extend(run_itemize(ocel, attr))

    with open(os.path.join(DATA_FOLDER, "items.json"), "w") as f:
        json.dump(items, f)

    return jsonify(status="success", redirect_url=url_for("show_items"))


@app.route("/aggregate", methods=["POST"])
def handle_aggregation():
    """
    Handle individual attribute aggregation and return discretization intervals.
    This endpoint performs aggregation and returns the intervals without page refresh.
    """
    data = request.get_json(force=True)
    attr = data.get("attribute", {})
    attribute_name = attr.get("attribute", {})
    agg_func = attr.get("aggregation", "")

    if not agg_func:
        return jsonify(status="failed", message="No aggregation specified"), 400

    try:
        ocel = load_ocel(os.path.join(DATA_FOLDER, "ocel.json"))
        _, items = handle_aggregate_attributes(ocel, attr)

        intervals = []

        for item in items:
            start = item.get("interval", "").get("start", -1)
            end = item.get("interval", "").get("end", -1)
            intervals.append(f"[{start}, {end}]")

        aggregated_attr = {
            "attribute": attr.get("attribute", ""),
            "original_attribute": attribute_name,
            "type": attr.get("type", ""),
            "qualifier": attr.get("qualifier", ""),
            "vals": [{"value": str(interval)} for interval in intervals],
            "aggregation": attr.get("aggregation", ""),
        }

        return jsonify(status="success", aggregated_attribute=aggregated_attr, message=f"Aggregation completed for {attr['attribute']}")

    except Exception as e:
        logger.exception("Failed to perform aggregation")
        return jsonify(status="failed", message=f"Aggregation failed: {str(e)}"), 500


@app.route("/items")
def show_items():
    """Render the page listing generated items for mining."""
    with open(os.path.join(DATA_FOLDER, "attributes.json")) as f:
        attrs = json.load(f)

    with open(os.path.join(DATA_FOLDER, "items.json")) as f:
        items = json.load(f)

    return render_template("items.html", items=items, attributes=attrs)


@app.route("/mine", methods=["POST"])
def mine():
    """
    Trigger mining routine: frequent itemsets, association rules,
    or classification rules based on the selected objective.
    """
    data = request.get_json(force=True)
    objective = data.get("objective", {}).get("name", {})
    params = data.get("objective", {}).get("parameters", {})
    ocel = load_ocel(os.path.join(DATA_FOLDER, "ocel.json"))

    with open(os.path.join(DATA_FOLDER, "items.json")) as f:
        items = json.load(f)

    transactions = transform_ocel(ocel, items)
    min_sup = float(params.get("min_sup", 0))
    frequent_itemsets = generate_frequent_itemsets(transactions, min_sup)

    if objective == "itemset":
        result = frequent_itemset_to_json(frequent_itemsets)
        target_file, endpoint = "itemset.json", "show_frequent_itemsets"
    else:
        min_conf = float(params.get("min_conf", 0))
        min_lift = float(params.get("min_lift", 0))
        rules_df = generate_association_rules(frequent_itemsets, min_conf, min_lift)
        if objective == "associationrule":
            result = association_rule_to_json(rules_df)
            target_file, endpoint = "association_rules.json", "show_association_rules"
        elif objective == "classificationrule":
            raw_targets = params.get("target", [])
            try:
                target_attrs = {json.loads(t).get("attribute", None) for t in raw_targets}
            except (TypeError, ValueError, KeyError) as e:
                return jsonify(status="failed", message=f"Invalid target format: {e}"), 400
            rules_df = generate_association_rules(frequent_itemsets, min_conf, min_lift)

            def all_consequents_in_target(consequents: frozenset) -> bool:
                attrs = {item_str.split("_", 1)[0] for item_str in consequents}
                return attrs == target_attrs

            filtered_df = pd.DataFrame(rules_df[rules_df["consequents"].apply(all_consequents_in_target)])
            result = association_rule_to_json(filtered_df)
            target_file, endpoint = "classification_rules.json", "show_classification_rules"
        else:
            return jsonify(status="failed", message="Unknown mining objective"), 400

    with open(os.path.join(DATA_FOLDER, target_file), "w") as f:
        json.dump(result, f)

    return jsonify(status="success", redirect_url=url_for(endpoint))


@app.route("/itemsets")
def show_frequent_itemsets():
    """Display mined frequent itemsets."""
    with open(os.path.join(DATA_FOLDER, "itemset.json")) as f:
        itemsets = json.load(f)

    return render_template("frequent_itemsets.html", frequent_itemsets=itemsets)


@app.route("/association_rules")
def show_association_rules():
    """Display mined association rules."""
    with open(os.path.join(DATA_FOLDER, "association_rules.json")) as f:
        rules = json.load(f)

    return render_template("association_rules.html", rules=rules)


@app.route("/classification_rules")
def show_classification_rules():
    """Display mined classification rules."""
    with open(os.path.join(DATA_FOLDER, "classification_rules.json")) as f:
        rules = json.load(f)

    return render_template("classification_rules.html", rules=rules)


@app.route("/download/ocel")
def download_ocel():
    """Download the exported OCEL JSON."""
    with open(os.path.join(DATA_FOLDER, "items.json")) as f:
        items = json.load(f)

    agg_items = {}

    for item in items:
        if item.get("aggregate", "") != "":
            attribute_name = item["attribute"]

            if attribute_name not in agg_items:
                agg_items[attribute_name] = {
                    "attribute": attribute_name,
                    "type": "EVENT",
                    "qualifier": item.get("qualifier"),
                    "aggregate": item.get("aggregate"),
                    "intervals": [],
                }

            if "interval" in item:
                agg_items[attribute_name]["intervals"].append(item["interval"])

    agg_items = list(agg_items.values())

    for item in agg_items:
        intervals = item.get("intervals", "")
        intervals = [(interval.get("start", 0), interval.get("end", 0)) for interval in item.get("intervals", "")]
        _apply_intervals_to_ocel(item, intervals, {})

    ocel = load_ocel(os.path.join(DATA_FOLDER, "ocel.json"))

    for df in (ocel.events, ocel.objects):
        temp_cols = [col for col in df.columns if col.startswith("__") and col.endswith("__")]
        for temp in temp_cols:
            base = temp.strip("_")
            df[base] = df[temp]
            df.drop(columns=[temp], inplace=True)

    save_ocel(ocel, os.path.join(DATA_FOLDER, "export_ocel.json"))

    return send_from_directory(directory=DATA_FOLDER, path="export_ocel.json", as_attachment=True, mimetype="application/json")


@app.route("/download/pattern/<pattern_name>")
def download_pattern(pattern_name):
    """
    Download a specific pattern JSON (itemset, association_rules, classification_rules).
    pattern_name should match the basename of the JSON file without extension.
    """
    filename = f"{pattern_name}.json"
    return send_from_directory(directory=DATA_FOLDER, path=filename, as_attachment=True, mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True)
