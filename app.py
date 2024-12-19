import os
import requests
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, jsonify, json
from ultralytics import YOLO
from dotenv import load_dotenv

# --- Load Variabel From .env ---
load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/predict"
CROP_FOLDER = "static/predict/crops/license-plate"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OCR_API_KEY = os.getenv("OCR_API_KEY")
SECRET_KEY = os.urandom(256)

# --- Initiate Flask App ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["CROP_FOLDER"] = CROP_FOLDER
app.config["SECRET_KEY"] = SECRET_KEY

# Setup Logging
logging.basicConfig(level=logging.INFO)

# --- Functions ---

# Check Extension (Allowed)
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# Function: Delete Images
def delete_images():
    try:
        # Define the folders to clean up
        folders_to_clean = [UPLOAD_FOLDER, RESULT_FOLDER, CROP_FOLDER]
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                # Remove files in the folder
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except PermissionError as e:
                        logging.error(f"Permission error while deleting file {file_path}: {e}")
                    except Exception as e:
                        logging.error(f"Error deleting file {file_path}: {e}")
                
                for subfolder in os.listdir(folder):
                    subfolder_path = os.path.join(folder, subfolder)
                    if os.path.isdir(subfolder_path):
                        # Remove files in the subfolder
                        for file in os.listdir(subfolder_path):
                            file_path = os.path.join(subfolder_path, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            except PermissionError as e:
                                logging.error(f"Permission error while deleting file {file_path}: {e}")
                            except Exception as e:
                                logging.error(f"Error deleting file {file_path}: {e}")

    except Exception as e:
        logging.error(f"Error deleting images: {e}")
        return jsonify({"error": str(e)}), 500

# Function: Image Prediction using You Only Look Once (YOLO)
def predict_image(image_path):
    model = YOLO("model/lnpr.pt")
    results = model.predict(
        source=image_path,
        save=True,
        save_crop=True,
        save_conf=True,
        stream=False,
        project="static"
    )
    return results

# Function: Extract Text From Image using OCR
def perform_ocr(image_path, api_key):
    payload = {
        "isOverlayRequired": False,
        "apikey": api_key,
        "language": "eng",
        "detectOrientation": True,
        "OCREngine": 2,
    }
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={image_path: f},
            data=payload
        )
    return response.content.decode()

# --- Flask App Routes ---

@app.route("/")
def index():
    delete_images()
    return render_template("index.html")

@app.route("/about-us")
def about_us():
    return render_template("about-us.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))
    
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    
    return redirect(url_for("show_image", filename=filename))

@app.route("/show/<filename>")
def show_image(filename):
    return render_template("read.html", filename=filename)

@app.route("/detect", methods=["POST"])
def detect():
    if request.method == "POST":
        image_file = request.form["file"]
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file)
        
        results = predict_image(image_path)
        if results:
            cropped_image_name = image_file
            return render_template("detect.html", image_predicted=cropped_image_name, img_name=image_file)
        else:
            return redirect(url_for("index"))

@app.route("/report", methods=["POST"])
def report():
    if request.method == "POST":
        image_file = request.form["img_name"]
        image_path = os.path.join(app.config["CROP_FOLDER"], image_file)
        
        ocr_result = perform_ocr(image_path, OCR_API_KEY)
        parsed_data = json.loads(ocr_result)
        
        if "ParsedResults" in parsed_data:
            parsed_text = parsed_data["ParsedResults"][0]["ParsedText"]
            parsed_text_split = parsed_text.split(" ")
            
            # OCR Processing
            if len(parsed_text_split) >= 3:
                plate_number = parsed_text_split[1]
                result_text = f"{parsed_text_split[0]} {plate_number} {parsed_text_split[2]}"
                
                curr_year = datetime.now().year
                if (curr_year % 2 == 0 and int(plate_number) % 2 == 0) or (curr_year % 2 == 1 and int(plate_number) % 2 == 1):
                    return render_template("report.html", txt_report=result_text, img_report=image_path, tilang_report="YES")
                else:
                    return render_template("report.html", txt_report=result_text, img_report=image_path, tilang_report="NO")
            else:
                return redirect(url_for("index"))
        else:
            return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
