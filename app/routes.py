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
import sys
import difflib
from models.images.extract_text_image import ExtractTextImage
from models.amazon.analyze_brand_title import AnalyzeBrandTitle
from managers.amazon.AmazonScraperManager import AmazonScraperManager


try:
    # Try setting local path (Windows dev)
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        # Set path for Linux (Docker)
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
except Exception as e:
    print(f"Error setting tesseract path: {e}")
    pass

# Load EAST text detector
# net = cv2.dnn.readNet("frozen_east_text_detection.pb")
# Process with EAST to find text regions first, then apply OCR only to those regions

from app import app


def is_similar(a, b, threshold=0.7):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

    
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
    

#Add default / index route
@app.route('/')
def index():
    print("Index route accessed")
    return "Welcome to the Scrapy API!"

# Analyze scraped result from search page in Amazon
@app.route('/analyze-search', methods=['POST'])
def analyze_search():
    data = request.get_json()
   
    #Check data is not empty and it is array
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Invalid data format. Expected a list.'}), 400
    
    # Check if the list is empty
    if not data:
        return jsonify({'error': 'No data provided.'}), 400
    
    # analyze the data
    analyzer = AnalyzeBrandTitle(data)
    
    # Call the analyze method
    analyzer.analyze()
    print(f"Analyzed data: {data}")
    
    # Retturn data as json
    return jsonify(data)

# Add search route
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request'}), 400
    
    url = data['url']
    
    return jsonify({'title': url})
    

@app.route('/predict-title', methods=['POST'])
def predict_title():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request'}), 400
    
    image_url = data['url']
    title = data['title']
    
    print(f"Received image URL and title: {image_url}, {title}")
    # title = extract_title_from_image_url(image_url)
    image_extractor = ExtractTextImage(image_path=image_url, title=title)
    response = image_extractor.extract_text_from_url()
    # response = extract_text_from_url(image_url, title)
    return jsonify({'title': response})


@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.get_json()
    start_url = data.get('start_url')
    
    if not start_url:
        return jsonify({'error': 'Missing "start_url" in request'}), 400
    
    print(f"Received start_url: {start_url}")
    
    scraper = AmazonScraperManager()
    
    # Use the AmazonScraperManager to scrape the product
    result = scraper.scrape_product(start_url)
    
    if result.get('success', False):
        return jsonify(result)
    else:
        # If there was an error, return with appropriate status code
        return jsonify(result), 500

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
