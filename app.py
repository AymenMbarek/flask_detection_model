from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import base64
import json
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def classify_error(area):
    if area < 50:
        return "Minor Error"
    elif area < 200:
        return "Moderate Error"
    else:
        return "Major Error"

def compare_and_annotate(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    original = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    error = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    min_height = min(original.shape[0], error.shape[0])
    min_width = min(original.shape[1], error.shape[1])
    original = cv2.resize(original, (min_width, min_height))
    error = cv2.resize(error, (min_width, min_height))

    difference = cv2.absdiff(original, error)
    thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]

    # Prepare the image where we'll draw the differences
    marked = cv2.cvtColor(error.copy(), cv2.COLOR_GRAY2BGR)

    # Find the contours of the differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    error_types = []  # To store the types of errors

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 255, 0), 1)
        error_type = classify_error(w*h)
        error_types.append({"type": error_type, "coordinates": (x, y), "size": (w, h)})

    # Convert marked image for display
    marked_display = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', marked_display)
    marked_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return marked_img_base64, error_types

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if files were uploaded
        if 'file1' not in request.files or 'file2' not in request.files:
            return "Please upload two files."

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Check if files are empty
        if file1.filename == '' or file2.filename == '':
            return "Please select two files."

        # Check file extensions
        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return "Allowed file types are: png, jpg, jpeg."

        # Save uploaded files
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(filepath1)
        file2.save(filepath2)

        # Compare and annotate images
        marked_img_base64, error_types = compare_and_annotate(filepath1, filepath2)

        return render_template('result.html', img1=filename1, img2=filename2, marked_img=marked_img_base64, error_types=error_types)

    return render_template('index.html')

@app.route('/save_json', methods=['POST'])
def save_json():
    data = request.form.get('results')
    if not data:
        return jsonify({"error": "No data received."}), 400

    try:
        results = json.loads(data)
        filename = f"detection_{int(time.time() * 1000)}.json"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "w") as json_file:
            json.dump(results, json_file, indent=4)
        return jsonify({"message": f"Detection results saved to {filename}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
