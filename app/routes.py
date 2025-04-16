from flask import Flask, request, jsonify
import subprocess
import pytesseract
import requests
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import shutil

try:
    # Try setting local path (Windows dev)
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # else:
        # Try setting path for Linux (Docker)
        # pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"    
except Exception:
    pass



from app import app

def extract_text_from_url(image_url, title_to_find):
    try:
        # Load image from URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_width = img_cv.shape[1]
        image_height = img_cv.shape[0]

        # Preprocessing
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite("debug_thresh.png", thresh)

        # Use pytesseract to get detailed data
        config = "--psm 6"
        data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

        found = False
        for i, text in enumerate(data["text"]):
            if text.strip().lower() == title_to_find.strip().lower():
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                height_percent = (h / image_height) * 100
                found = True

                print(f"Title: '{text}'")
                print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
                print(f"Height percent of image: {height_percent:.2f}%")

                return {
                    "text": text,
                    "bounding_box": (x, y, w, h),
                    "height_percent": height_percent
                }

        if not found:
            return {"error": "Title not found in image."}

    except Exception as e:
        return {"error": f"Error processing image: {e}"}
    
def extract_title_from_image_url(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        print(f"Image URL: {image_url}")
        print(f"Image size: {img.size}")
        
        text = pytesseract.image_to_string(img_cv)
        print(f"Extracted text: {text}")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        possible_title = max(lines, key=len) if lines else "Unknown Product"
        return possible_title
    except Exception as e:
        return f"Error processing image: {str(e)}"
    
@app.route("/debug-tesseract-full")
def debug_tesseract_full():
    import subprocess, os
    return {
        "env_path": os.environ.get("PATH"),
        "list_usr_bin": subprocess.check_output(["ls", "-la", "/usr/bin"]).decode("utf-8"),
        "tesseract_installed": subprocess.check_output(["apt", "list", "--installed", "tesseract*"]).decode("utf-8")
    }

#Add default / index route
@app.route('/')
def index():
    print("Index route accessed")
    return "Welcome to the Scrapy API!"

@app.route('/predict-title', methods=['POST'])
def predict_title():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request'}), 400
    
    image_url = data['url']
    title = data['title']
    
    print(f"Received image URL: {image_url}")
    # title = extract_title_from_image_url(image_url)
    response = extract_text_from_url(image_url, title)
    return jsonify({'title': response})


@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.get_json()
    start_url = data.get('start_url', 'https://www.amazon.com/TOZO-Cancelling-Waterproof-Bluetooth-Headphones/dp/B0DG8NMPSH')
    
    print(f"Received start_url: {start_url}")

    # Define the path to your Scrapy project
    scrapy_project_path = os.path.join(os.getcwd(), 'scrapy_spider')

    # Define the absolute path for the output file
    output_file_path = os.path.join(os.getcwd(), 'output.json')

    # Construct the Scrapy command
    command = [
        'scrapy', 'crawl', 'example_spider',
        '-a', f'start_url={start_url}',
        '-o', output_file_path
    ]

    try:
        # Run the Scrapy command
        result = subprocess.run(
            command,
            cwd=scrapy_project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True  # This will raise an exception for non-zero exit codes
        )

        # Check if the output file exists
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as file:
                data = file.read()
            return jsonify({'data': data})
        else:
            return jsonify({'error': 'Output file not found.'}), 500

    except subprocess.CalledProcessError as e:
        # Handle errors in the subprocess
        return jsonify({'error': f'Scrapy command failed: {e.stderr}'}), 500
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
