import json
import os
import warnings

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from pandas.errors import SettingWithCopyWarning
from werkzeug.utils import secure_filename

from src.aggregation import handle_aggregate_attributes
from src.mining import (
    association_rule_to_json,
    frequent_itemset_to_json,
    generate_association_rules,
    generate_frequent_itemsets,
    run_itemize,
    transform_ocel,
)
from src.preprocessing import allowed_file, get_attributes, load_ocel, save_ocel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
ALLOWED_EXTENSIONS = {"json", "sqlite"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "key"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.template_filter("file_extension")
def file_extension(filename):
    return filename.split(".")[1].upper() if "." in filename else ""


@app.route("/", methods=["GET", "POST"])
def upload_file():
    existing_files = []
    upload_folder = app.config["UPLOAD_FOLDER"]
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            if f.split(".")[1] in ALLOWED_EXTENSIONS:
                existing_files.append(f)

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if not file:
                raise Exception("No OCEL selected")

            if (
                file
                and file.filename
                and allowed_file(file.filename, str(ALLOWED_EXTENSIONS))
            ):
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                try:
                    ocel = load_ocel(file_path)
                    save_ocel(ocel, f"{DATA_FOLDER}/ocel.json")
                    attributes = get_attributes(ocel)

                    return render_template(
                        "select.html", filename=filename, attributes=attributes
                    )
                except Exception as e:
                    flash(f"Fehler: {str(e)}")
                    return redirect(request.url)

        elif "existing_file" in request.form:
            filename = request.form["existing_file"]
            file_path = os.path.join(upload_folder, filename)
            if os.path.exists(file_path):
                try:
                    ocel = load_ocel(file_path)
                    save_ocel(ocel, f"{DATA_FOLDER}/ocel.json")
                    attributes = get_attributes(ocel)

                    return render_template(
                        "select.html", filename=filename, attributes=attributes
                    )
                except Exception as e:
                    flash(f"Fehler: {str(e)}")
                    return redirect(request.url)
            else:
                flash("Datei existiert nicht mehr")

        return redirect(request.url)

    return render_template("index.html", existing_files=existing_files)


@app.route("/process", methods=["POST"])
def process_attributes():
    data = request.get_json()
    with open("data/attributes.json", "w") as f:
        json.dump(data["attributes"], f)

    ocel = load_ocel("data/ocel.json")

    items = []

    for attr in data["attributes"]:
        if attr["aggregation"] != "":
            ocel, item = handle_aggregate_attributes(ocel, attr)
            save_ocel(ocel, "data/ocel.json")
            items.extend(item)
        elif attr["selected"]:
            items.extend(run_itemize(ocel, attr))

    with open("data/items.json", "w") as f:
        json.dump(items, f)

    return jsonify({"status": "success", "redirect_url": url_for("show_items")})


@app.route("/items")
def show_items():
    with open("data/attributes.json", "r") as f:
        attributes = json.load(f)

    with open("data/items.json", "r") as f:
        items = json.load(f)
    return render_template(
        "items.html", items=items, attributes=attributes
    )


@app.route("/mine", methods=["POST"])
def mine():
    data = request.get_json()
    ocel = load_ocel("data/ocel.json")

    with open("data/items.json", "r") as f:
        items = json.load(f)

    objective = data["objective"]["name"]
    parameters = data["objective"]["parameters"]

    match objective:
        case "itemset":
            transactions = transform_ocel(ocel, items)
            frequent_itemsets = generate_frequent_itemsets(
                transactions, float(parameters["min_sup"])
            )
            frequent_itemsets = frequent_itemset_to_json(
                frequent_itemsets
            )
            with open("data/itemset.json", "w") as f:
                json.dump(frequent_itemsets, f)
            return jsonify(
                {
                    "status": "success",
                    "redirect_url": url_for("show_frequent_itemsets"),
                }
            )

        case "associationrule":
            transactions = transform_ocel(ocel, items)
            frequent_itemsets = generate_frequent_itemsets(
                transactions, float(parameters["min_sup"])
            )
            association_rules = generate_association_rules(
                frequent_itemsets, float(parameters["min_conf"]), float(parameters["min_lift"])
            )
            association_rules = association_rule_to_json(
                association_rules
            )
            with open("data/association_rules.json", "w") as f:
                json.dump(association_rules, f)
            return jsonify(
                {
                    "status": "success",
                    "redirect_url": url_for("show_association_rules"),
                }
            )

        case "classificationrule":
            transactions = transform_ocel(ocel, items)
            frequent_itemsets = generate_frequent_itemsets(
                transactions, float(parameters["min_sup"])
            )
            association_rules = generate_association_rules(
                frequent_itemsets, float(parameters["min_conf"]), float(parameters["min_lift"])
            )
            association_rules = association_rule_to_json(
                association_rules
            )
            rules = []
            target = json.load(parameters["target"])
            for rule in association_rules:
                if (
                    len(rule["consequents"]) == 1
                    and rule["consequents"][0]["attribute"] == target["attribute"]
                ):
                    rules.append(rule)
            with open("data/classification_rules.json", "w") as f:
                json.dump(rules, f)
            return jsonify(
                {
                    "status": "success",
                    "redirect_url": url_for("show_classification_rules"),
                }
            )

    return jsonify(
        {
            "status": "failed",
        }
    )


@app.route("/itemsets")
def show_frequent_itemsets():
    with open("data/itemset.json", "r") as f:
        itemsets = json.load(f)
    return render_template("frequent_itemsets.html", frequent_itemsets=itemsets)


@app.route("/association_rules")
def show_association_rules():
    with open("data/association_rules.json", "r") as f:
        rules = json.load(f)
    return render_template("association_rules.html", rules=rules)


@app.route("/classification_rules")
def show_classification_rules():
    with open("data/classification_rules.json", "r") as f:
        rules = json.load(f)
    return render_template("classification_rules.html", rules=rules)


if __name__ == "__main__":
    app.run(debug=True)
