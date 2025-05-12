import os
from flask import Flask, flash, request, redirect, render_template, session
from werkzeug.utils import secure_filename
import helper
import warnings
from pandas.errors import SettingWithCopyWarning

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
                ocel = helper.load_ocel(file_path)
                attributes = helper.get_attributes(ocel)
                
                return render_template('results.html', 
                                      filename=filename,
                                      attributes=attributes)
            except Exception as e:
                flash(f'Fehler: {str(e)}')
                return redirect(request.url)

    return render_template('index.html')

@app.route('/get_related_attributes')
def get_related_attributes():
    attribute = request.args.get('attribute')

    file_path = session.get('current_ocel')
    if not file_path:
        flash('Fehler beim Laden des OCEL')
        return redirect(request.url)
    ocel = helper.load_ocel(file_path)

    return helper.get_related_attributes(ocel=ocel, attribute=attribute) # type:ignore

@app.route('/process', methods=['POST'])
def process_attributes():
    algorithm = request.form.get('algorithm')
    selected = request.form.getlist('not_numeric')
    numeric = request.form.getlist('numeric')
    related = request.form.getlist(f'related_1[]')

    file_path = session.get('current_ocel')
    if not file_path:
        flash('Fehler beim Laden des OCEL')
        return redirect(request.url)
    ocel = helper.load_ocel(file_path)

    results = {
        "attributes": []
    }

    match algorithm:
        case "equal_freq":
            for i, attribute in enumerate(numeric):
                params = request.form.get(f"params[{i}][bins]")
                results["attributes"].append(helper.run_equal_frequency_binning(ocel, attribute, params, related)) # type:ignore
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

if __name__ == '__main__':
    app.run(debug=True)
