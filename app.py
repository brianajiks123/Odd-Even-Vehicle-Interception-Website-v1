import os, logging, json, requests
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/predict"
CROP_FOLDER = "static/predict/crops/license-plate"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OCR_API_KEY = os.getenv("OCR_API_KEY") or ""
SECRET_KEY = os.getenv("SECRET_KEY") or ""

# --- Flask App Setup ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["CROP_FOLDER"] = CROP_FOLDER
app.config["SECRET_KEY"] = SECRET_KEY

if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in .env file")

if not OCR_API_KEY:
    raise ValueError("OCR_API_KEY must be set in .env file")

for folder in [UPLOAD_FOLDER, RESULT_FOLDER, CROP_FOLDER]:
    Path(folder).mkdir(parents = True, exist_ok = True)

# --- Logging Setup ---
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()

        self.stream = open(self.stream.fileno(), mode = 'w', encoding = 'utf-8', errors = 'replace')

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    handlers = [
        UTF8StreamHandler(),
        logging.FileHandler("app.log", encoding = 'utf-8')
    ]
)

logger = logging.getLogger(__name__)

# --- Load YOLO Model ---
try:
    YOLO_MODEL = YOLO("model/lnpr.pt")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

# --- Utility Functions ---

def allowed_file(filename):
    """Check if the file extension is allowed."""
    if filename is None:
        return False

    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def delete_images():
    """Delete all files in upload, result, and crop folders."""
    try:
        folders_to_clean = [UPLOAD_FOLDER, RESULT_FOLDER, CROP_FOLDER]

        for folder in folders_to_clean:
            folder_path = Path(folder)

            if not folder_path.exists():
                continue

            for item in folder_path.rglob("*"):
                if item.is_file():
                    try:
                        item.unlink()
                        logger.info(f"Deleted file: {item}")
                    except PermissionError as e:
                        logger.error(f"Permission error deleting {item}: {e}")
                    except Exception as e:
                        logger.error(f"Error deleting {item}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during image cleanup: {e}")

        return {"error": str(e)}

def predict_image(image_path):
    """Perform YOLO prediction on the given image and return results and predicted image path."""
    try:
        Path(CROP_FOLDER).mkdir(parents = True, exist_ok = True)

        results = YOLO_MODEL.predict(
            source = image_path,
            save = True,
            save_crop = True,
            save_conf = True,
            stream = False,
            project = RESULT_FOLDER,
            name = ".",
            exist_ok = True,
            save_txt = False
        )

        predicted_image_path = None

        for result in results:
            if result.boxes and result.boxes.xyxy.shape[0] > 0:
                if result.save_dir is None:
                    logger.warning("save_dir is None, skipping crop handling")

                    continue

                base_filename = Path(image_path).stem
                save_dir = Path(result.save_dir)

                for ext in ALLOWED_EXTENSIONS:
                    candidate_path = save_dir / f"{base_filename}{ext}"

                    if candidate_path.exists():
                        predicted_image_path = candidate_path

                        break

                if predicted_image_path:
                    logger.info(f"Predicted image saved at: {predicted_image_path}")
                else:
                    logger.warning(f"Predicted image not found for {base_filename} in {save_dir}")

                crop_dir = Path(result.save_dir) / "crops" / "license-plate"

                if crop_dir.exists():
                    for crop_file in crop_dir.glob("*.jpg"):
                        target_path = Path(CROP_FOLDER) / crop_file.name

                        crop_file.rename(target_path)
                        logger.info(f"Moved crop file to: {target_path}")

                    try:
                        crop_dir.rmdir()
                        (crop_dir.parent).rmdir()
                        logger.info(f"Removed empty folder: {crop_dir}")
                    except OSError:
                        pass

        logger.info(f"Successfully predicted image: {image_path}")

        relative_path = str(predicted_image_path.relative_to("static")).replace("\\", "/") if predicted_image_path else None

        return results, relative_path
    except Exception as e:
        logger.error(f"Error predicting image {image_path}: {e}")

        return None, None

def perform_ocr(image_path, api_key):
    """Perform OCR on the given image using OCR.Space API."""
    if not api_key:
        logger.error("OCR API key is missing")

        return "Error: API key is required"
    if not image_path:
        logger.error("Image path is missing")

        return "Error: Image path is required"
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")

        return f"Error: Image file not found at {image_path}"
    if not allowed_file(image_path):
        logger.error(f"Invalid image format: {image_path}")

        return f"Error: Invalid image format. Supported formats: {ALLOWED_EXTENSIONS}"

    payload = {
        "isOverlayRequired": False,
        "apikey": api_key,
        "language": "auto",
        "detectOrientation": True,
        "scale": True,
        "OCREngine": 2,
    }

    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files = {Path(image_path).name: f},
                data = payload,
                timeout = 30
            )

            response.raise_for_status()

            result = response.json()

            if result.get("IsErroredOnProcessing", True):
                error_message = result.get("ErrorMessage", ["Unknown error"])[0]

                logger.error(f"OCR processing error: {error_message}")

                return f"OCR Error: {error_message}"

            logger.info(f"OCR successful for {image_path}")

            return result
    except FileNotFoundError:
        logger.error(f"File not found: {image_path}")

        return f"Error: Unable to open file at {image_path}"
    except requests.Timeout:
        logger.error("OCR API request timed out")

        return "Error: Request timed out"
    except requests.ConnectionError:
        logger.error("Failed to connect to OCR API")

        return "Error: Failed to connect to the API"
    except requests.HTTPError as e:
        logger.error(f"HTTP error in OCR request: {e}")

        return f"HTTP Error: {str(e)}"
    except requests.RequestException as e:
        logger.error(f"Request error in OCR: {e}")

        return f"Request Error: {str(e)}"
    except json.JSONDecodeError:
        logger.error("Invalid JSON response from OCR API")

        return "Error: Invalid JSON response from API"
    except Exception as e:
        logger.error(f"Unexpected error in OCR: {e}")

        return f"Unexpected Error: {str(e)}"

# --- Flask Routes ---

@app.route("/")
def index():
    """Render the index page and clean up old images."""
    error = delete_images()

    if error:
        return jsonify(error), 500

    return render_template("index.html")

@app.route("/about-us")
def about_us():
    """Render the about-us page."""
    return render_template("about-us.html")

@app.route("/upload", methods = ["POST"])
def upload():
    """Handle file upload and validation."""
    if "file" not in request.files:
        logger.warning("No file part in request")

        return redirect(url_for("index"))

    file = request.files["file"]

    if not file or file.filename == "" or not allowed_file(file.filename):
        logger.warning(f"Invalid file: {file.filename if file else 'None'}")

        return redirect(url_for("index"))
    if not file.mimetype.startswith("image/"):
        logger.warning(f"Invalid MIME type: {file.mimetype}")

        return redirect(url_for("index"))

    try:
        file.stream.seek(0)

        img = Image.open(file.stream)

        img.verify()
        file.stream.seek(0)
    except Exception as e:
        logger.error(f"Invalid image file: {e}")

        return redirect(url_for("index"))

    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
    file_path = Path(app.config["UPLOAD_FOLDER"]) / filename

    file.save(file_path)
    logger.info(f"File uploaded: {file_path}")

    return redirect(url_for("show_image", filename = filename))

@app.route("/show/<filename>")
def show_image(filename):
    """Display the uploaded image."""
    if not filename or not allowed_file(filename) or not (Path(app.config["UPLOAD_FOLDER"]) / filename).exists():
        logger.warning(f"Invalid or missing file: {filename}")

        return redirect(url_for("index"))

    return render_template("read.html", filename = filename)

@app.route("/detect", methods = ["POST"])
def detect():
    """Perform YOLO detection on the uploaded image."""
    image_file = request.form.get("file")

    if not image_file or not allowed_file(image_file):
        logger.warning(f"Invalid or missing image file: {image_file}")

        return redirect(url_for("index"))

    image_path = Path(app.config["UPLOAD_FOLDER"]) / image_file

    if not image_path.exists():
        logger.warning(f"Image file not found: {image_file}")

        return redirect(url_for("index"))

    logger.info(f"Image path: {image_path}")

    results, predicted_image = predict_image(str(image_path))

    if not results or not predicted_image:
        logger.error(f"Prediction failed for {image_path}")

        return redirect(url_for("index"))

    logger.info(f"Rendering detect.html with image_predicted: {predicted_image}")

    crop_dir = Path(app.config["CROP_FOLDER"])
    crop_files = list(crop_dir.glob("*.jpg"))
    crop_filename = crop_files[0].name if crop_files else image_file

    return render_template("detect.html", image_predicted = predicted_image, img_name = crop_filename)

@app.route("/report", methods = ["POST"])
def report():
    """Generate report based on OCR results."""
    image_file = request.form.get("img_name")

    if not image_file or not allowed_file(image_file):
        logger.warning(f"Invalid or missing crop image: {image_file}")

        return redirect(url_for("index"))

    image_path = Path(app.config["CROP_FOLDER"]) / image_file

    if not image_path.exists():
        logger.warning(f"Crop image not found: {image_file}")

        return redirect(url_for("index"))

    ocr_result = perform_ocr(str(image_path), OCR_API_KEY)

    if isinstance(ocr_result, str):
        logger.error(f"OCR failed: {ocr_result}")

        return redirect(url_for("index"))

    parsed_data = ocr_result

    if "ParsedResults" not in parsed_data or not parsed_data["ParsedResults"]:
        logger.error("No valid OCR results found")

        return redirect(url_for("index"))

    parsed_text = parsed_data["ParsedResults"][0]["ParsedText"]

    try:
        parsed_text_split = parsed_text.split()

        if len(parsed_text_split) < 3:
            logger.warning(f"Insufficient OCR text segments: {parsed_text}")

            return redirect(url_for("index"))

        plate_number = parsed_text_split[1]

        if not plate_number.isdigit():
            logger.warning(f"Invalid plate number format: {plate_number}")

            return render_template(
                "report.html",
                txt_report = f"{parsed_text_split[0]} {plate_number} {parsed_text_split[2]}",
                img_report = str(image_path.relative_to("static")).replace("\\", "/"),
                tilang_report = "UNKNOWN (Invalid plate number format)"
            )

        result_text = f"{parsed_text_split[0]} {plate_number} {parsed_text_split[2]}"
        curr_year = datetime.now().year
        is_violation = (curr_year % 2 == 0 and int(plate_number) % 2 == 0) or (curr_year % 2 == 1 and int(plate_number) % 2 == 1)
        tilang_report = "YES" if is_violation else "NO"

        logger.info(f"Report generated: {result_text}")
        logger.info(f"Loading image for report: {image_path}")

        return render_template(
            "report.html",
            txt_report = result_text,
            img_report = str(image_path.relative_to("static")).replace("\\", "/"),
            tilang_report = tilang_report
        )
    except (ValueError, IndexError) as e:
        logger.error(f"Error processing OCR results: {str(e)}")

        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug = False, host = "0.0.0.0", port = 5000)
