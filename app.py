import os
from flask import Flask, flash, request, redirect, render_template, session, url_for, jsonify
from werkzeug.utils import secure_filename
import helper
import warnings
from pandas.errors import SettingWithCopyWarning
import json

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json', 'sqlite'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'key'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.template_filter('file_extension')
def file_extension(filename):
    return filename.split(".")[1].upper() if '.' in filename else ''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    existing_files = []
    upload_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            if f.split(".")[1] in ALLOWED_EXTENSIONS:
                existing_files.append(f)

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('Keine Datei ausgewählt')
                return redirect(request.url)
            
            if file and helper.allowed_file(file.filename, str(ALLOWED_EXTENSIONS)):
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                session['current_ocel'] = file_path
                try:
                    ocel = helper.load_ocel(file_path)
                    attributes = helper.get_attributes(ocel)
        
                    return render_template('select.html', 
                                        filename=filename,
                                        attributes=attributes)
                except Exception as e:
                    flash(f'Fehler: {str(e)}')
                    return redirect(request.url)

        elif 'existing_file' in request.form:
            filename = request.form['existing_file']
            file_path = os.path.join(upload_folder, filename)
            if os.path.exists(file_path):
                session['current_ocel'] = file_path
                try:
                    ocel = helper.load_ocel(file_path)
                    attributes = helper.get_attributes(ocel)

                    return render_template('select.html', 
                                        filename=filename,
                                        attributes=attributes)
                except Exception as e:
                    flash(f'Fehler: {str(e)}')
                    return redirect(request.url)
            else:
                flash('Datei existiert nicht mehr')
        
        return redirect(request.url)

    return render_template('index.html', existing_files=existing_files)


@app.route('/process', methods=['POST'])
def process_attributes():

    data = request.get_json()

    selected = [attribute for attribute in data['attributes'] if attribute['selected']]
    numeric = [attribute for attribute in selected if attribute['numeric']]
    not_numeric = [attribute for attribute in selected if not attribute['numeric']]

    file_path = session.get('current_ocel')
    if not file_path:
        flash('Fehler beim Laden des OCEL')
        return redirect(request.url)
    ocel = helper.load_ocel(file_path)

    items = []

    for attribute in numeric:
        items.extend(helper.run_itemize(ocel, attribute, data["algorithm"]))

    for attribute in not_numeric:
        items.extend(helper.not_numeric(ocel, attribute))

    with open("items.json", "w") as f:
        json.dump(items, f)

    return jsonify({"status": "success", "redirect_url": url_for('show_items')})

@app.route('/items')
def show_items():
    with open("items.json", "r") as f:
        items = json.load(f)
    return render_template('items.html', items=items)

@app.route('/mine', methods=['POST'])
def mine():

    data = request.get_json()


    with open("items.json", "r") as f:
        items = json.load(f)
    
    objective = data["objective"]["name"]
    parameters = data["objective"]["parameters"]

    match objective:
        case "itemset":
            pass
            


    return jsonify({"status": "success", "redirect_url": url_for('show_objective')})

@app.route('/result')
def show_objective():
    return render_template('frequent_itemsets.html')




if __name__ == '__main__':
    app.run(debug=True)
