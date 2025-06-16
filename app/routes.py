from flask import Flask, Blueprint, request, jsonify
# from models.amazon.analyze_brand_title import ProductTitleExtractor
from models.amazon.analyze_image_texts import AnalyzeImageTexts
from managers.amazon.AmazonScraperManager import AmazonScraperManager
from managers.sainsburys.SainsburysScraperManager import SainsburysScraperManager
# from models.amazon.title_predictor_demo import TitlePredictorDemo
from models.amazon.title_prediction_service import TitlePredictionService
from database.supabase import SupabaseManager
import asyncio

from app import app

from routes.route_model import model_bp  # Import your blueprint

# Register blueprint with /model prefix
app.register_blueprint(model_bp, url_prefix='/model')

# Create a service instance
title_service = TitlePredictionService()   

#Add default / index route
@app.route('/')
def index():
    return "Welcome to the Scrapy API!"


# Analyze scraped result from search page in Amazon
@app.route('/analyze-search', methods=['POST'])
def analyze_search():
    data = request.get_json()
   
    #Check data is not empty and it have insertedId and products
    if not data or 'insertedId' not in data or 'products' not in data:
        return jsonify({'error': 'Missing "insertedId" or "products" in request'}), 400
    
    products = data['products']
    insertedId = data['insertedId']
    
    # Check if the list is empty
    if not products:
        return jsonify({'error': 'No data provided.'}), 400
    
    # analyze the data
    analyzer = AnalyzeImageTexts(products)
    
    processed_data = analyzer.analyze()
    
    # Add title predictions to each item in processed_data
    for item in processed_data:
        # Make prediction with self-learning model
        prediction = title_service.predict_title(item)
        
        # Add prediction to item
        item['predicted_title'] = prediction.get('predicted_title')
        item['prediction_metrics'] = {
            'height_percent': prediction.get('height_percent'),
            'contrast_ratio': prediction.get('average_contrast_ratio'),
            'visible_ratio': prediction.get('visible_ratio')
        }
    
    response = {
        "items": processed_data,
        "learning_stats": title_service.get_learning_stats()
    }
    
    supabase = SupabaseManager()
    # Insert the processed data into Supabase
    supabase.update_data("search", response, insertedId, "python_results")
    
    return jsonify(response)

# Add search route
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request'}), 400
    
    url = data['url']
    
    return jsonify({'title': url})

# Add Sainsburys crawl for product page
@app.route('/sainsburys-product-page', methods=['POST'])
def sainsburys_product_page():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request'}), 400
    
    url = data['url']
    
    # scraper = SainsburysScraperManager()
    
    crawler = SainsburysScraperManager(url)
    asyncio.run(crawler.fetch())  # Run the async crawler
    try:
        result = crawler.get_result()
        print(f"Result from SainsburysScraperManager: {result}")  # Debugging line
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'No data found for the provided URL'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
    

@app.route('/predict-title', methods=['POST'])
def predict_title():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" in request'}), 400
    
    image_url = data['url']
    title = data['title']
    
    return jsonify({'title': title, 'url': image_url})


@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.get_json()
    start_url = data.get('start_url')
    
    if not start_url:
        return jsonify({'error': 'Missing "start_url" in request'}), 400
    
    scraper = AmazonScraperManager()
    
    # Use the AmazonScraperManager to scrape the product
    result = scraper.scrape_product(start_url)
    
    if result.get('success', False):
        return jsonify(result)
    else:
        # If there was an error, return with appropriate status code
        return jsonify(result), 500
