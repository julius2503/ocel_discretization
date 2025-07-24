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
    session,
    url_for,
)
from pandas.errors import SettingWithCopyWarning
from werkzeug.utils import secure_filename

from src import helper, mining

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"json", "sqlite"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "key"

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
                and helper.allowed_file(file.filename, str(ALLOWED_EXTENSIONS))
            ):
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                session["current_ocel"] = file_path
                try:
                    ocel = helper.load_ocel(file_path)
                    attributes = helper.get_attributes(ocel)

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
                session["current_ocel"] = file_path
                try:
                    ocel = helper.load_ocel(file_path)
                    attributes = helper.get_attributes(ocel)

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

    selected = [
        attribute for attribute in data["attributes"] if attribute["selected"]
    ]
    session["selected"] = selected
    numeric = [attribute for attribute in selected if attribute["numeric"]]
    not_numeric = [
        attribute for attribute in selected if not attribute["numeric"]
    ]

    file_path = session.get("current_ocel")
    if not file_path:
        flash("Fehler beim Laden des OCEL")
        return redirect(request.url)
    ocel = helper.load_ocel(file_path)

    items = []

    for attribute in numeric:
        items.extend(helper.run_itemize(ocel, attribute, data["algorithm"]))

    for attribute in not_numeric:
        items.extend(helper.not_numeric(ocel, attribute))

    with open("data/items.json", "w") as f:
        json.dump(items, f)

    return jsonify({"status": "success", "redirect_url": url_for("show_items")})


@app.route("/items")
def show_items():
    with open("data/items.json", "r") as f:
        items = json.load(f)
    return render_template(
        "items.html", items=items, attributes=session.get("selected")
    )


@app.route("/mine", methods=["POST"])
def mine():
    data = request.get_json()

    file_path = session.get("current_ocel")
    if not file_path:
        flash("Fehler beim Laden des OCEL")
        return redirect(request.url)
    ocel = helper.load_ocel(file_path)

    with open("data/items.json", "r") as f:
        items = json.load(f)

    objective = data["objective"]["name"]
    parameters = data["objective"]["parameters"]

    match objective:
        case "itemset":
            transactions = mining.tranform_ocel(ocel, items)
            frequent_itemsets = mining.generate_frequent_itemsets(
                transactions, float(parameters["min_sup"])
            )
            frequent_itemsets = mining.frequent_itemset_to_json(
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
            transactions = mining.tranform_ocel(ocel, items)
            frequent_itemsets = mining.generate_frequent_itemsets(
                transactions, float(parameters["min_sup"])
            )
            association_rules = mining.generate_association_rules(
                frequent_itemsets, float(parameters["min_lift"])
            )
            association_rules = mining.association_rule_to_json(
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
            transactions = mining.tranform_ocel(ocel, items)
            frequent_itemsets = mining.generate_frequent_itemsets(
                transactions, float(parameters["min_sup"])
            )
            association_rules = mining.generate_association_rules(
                frequent_itemsets, float(parameters["min_lift"])
            )
            association_rules = mining.association_rule_to_json(
                association_rules
            )
            rules = []
            attribute, _, _ = parameters["target"].split("-")
            for rule in association_rules:
                if (
                    len(rule["consequents"]) == 1
                    and rule["consequents"][0]["attribute"] == attribute
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
