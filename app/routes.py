from flask import Flask, request, jsonify
import subprocess
import os
import json
# from models.amazon.analyze_brand_title import ProductTitleExtractor
from models.amazon.analyze_image_texts import AnalyzeImageTexts
from managers.amazon.AmazonScraperManager import AmazonScraperManager
from models.amazon.sefl_title_predictor import SelfLearningTitlePredictor
from models.amazon.title_predictor_demo import TitlePredictorDemo


from app import app

class TitlePredictionService:
    def __init__(self, memory_file="title_predictor_memory.json"):
        self.memory_file = memory_file
        self.stats = {
            "predictions": 0,
            "feedback_received": 0,
            "avg_similarity": 0.0
        }
    
    def predict_title(self, data):
        """Make a prediction for a single item"""
        predictor = SelfLearningTitlePredictor(data, self.memory_file)
        prediction = predictor.predict()
        
        self.stats["predictions"] += 1
        return prediction
    
    def batch_predict(self, items):
        """Process multiple items and make predictions"""
        results = []
        
        for item in items:
            prediction = self.predict_title(item)
            
            # Add the prediction to results
            results.append({
                "item_id": item.get("id", "unknown"),
                "brand": item.get("brand", ""),
                "original_title": item.get("title", ""),
                "predicted_title": prediction.get("predicted_title", ""),
                "confidence_metrics": {
                    "height_percent": prediction.get("height_percent", 0),
                    "contrast_ratio": prediction.get("average_contrast_ratio", 0),
                    "visible_ratio": prediction.get("visible_ratio", 0)
                }
            })
        
        return results
    
    def provide_feedback(self, item_id, original_prediction, corrected_title):
        """
        Process feedback for an item
        
        Args:
            item_id: ID of the item to provide feedback for
            original_prediction: Original prediction made by the system
            corrected_title: Corrected title provided by user
            
        Returns:
            Dictionary with feedback results
        """
        try:
            # Try to load the temp item data
            temp_file = f"temp_data_{item_id}.json"
            
            if not os.path.exists(temp_file):
                return {
                    "success": False,
                    "error": f"Item data not found for ID: {item_id}"
                }
            
            with open(temp_file, 'r') as f:
                item_data = json.load(f)
            
            # Create predictor with the loaded data
            predictor = SelfLearningTitlePredictor(item_data, self.memory_file)
            
            # Explicitly set the item attribute
            predictor.item = item_data
            
            # If original_prediction is provided, manually set it as the previous prediction
            if original_prediction:
                predictor.previous_predictions = {
                    'predicted_title': original_prediction
                }
            
            # Process the feedback
            result = predictor.give_feedback(corrected_title)
            
            if not result:
                return {
                    "success": False,
                    "error": "Failed to process feedback - internal predictor error"
                }
            
            return {
                "success": True,
                "similarity": result.get("similarity", 0),
                "item_id": item_id
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Exception occurred: {str(e)}"
            }
    
    def get_learning_stats(self):
        """Get stats about the learning progress"""
        # Create a temporary predictor just to access memory
        temp_predictor = SelfLearningTitlePredictor({}, self.memory_file)
        
        top_patterns = dict(sorted(
            temp_predictor.successful_patterns.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        return {
            "predictions_made": self.stats["predictions"],
            "feedback_received": self.stats["feedback_received"],
            "average_similarity": round(self.stats["avg_similarity"], 2),
            "top_learned_patterns": top_patterns,
            "current_parameters": temp_predictor.params
        }

# Create a service instance
title_service = TitlePredictionService()    


#Add default / index route
@app.route('/')
def index():
    return "Welcome to the Scrapy API!"


@app.route('/predict', methods=['POST'])
def predict_titles():
    """API endpoint for predicting titles"""
    if not request.json or 'items' not in request.json:
        return jsonify({"error": "Invalid request"}), 400
    
    items = request.json['items']
    
    # For each item, temporarily store the data for feedback purposes
    for item in items:
        item_id = item.get("id", f"temp_{hash(json.dumps(item))}")
        with open(f"temp_data_{item_id}.json", 'w') as f:
            json.dump(item, f)
    
    # Process items and make predictions
    results = title_service.batch_predict(items)
    
    return jsonify({
        "results": results,
        "stats": title_service.get_learning_stats()
    })

@app.route('/feedback', methods=['POST'])
def provide_feedback():
    """API endpoint for providing feedback"""
    if not request.json:
        return jsonify({"error": "Invalid request"}), 400
    
    item_id = request.json.get('item_id')
    original_prediction = request.json.get('original_prediction')
    corrected_title = request.json.get('corrected_title')
    
    print(f"Feedback received for item {item_id}: {corrected_title}")
    
    if not item_id or not corrected_title:
        return jsonify({"error": "Missing required fields"}), 400
    
    result = title_service.provide_feedback(item_id, original_prediction, corrected_title)
    
    print(f"Feedback processing result: {result}")
    
    if not result.get("success"):
        return jsonify({"error": result.get("error", "Failed to process feedback")}), 400
    
    return jsonify({
        "success": True,
        "stats": title_service.get_learning_stats()
    })

@app.route('/learning-stats', methods=['GET'])
def get_learning_stats():
    """API endpoint for getting learning stats"""
    return jsonify(title_service.get_learning_stats())

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
    # analyzer = ProductTitleExtractor(data)
    analyzerv2 = AnalyzeImageTexts(data)
    
    # processed_data = analyzer.analyze()
    processed_data = analyzerv2.analyze()
    
     # Add title predictions to each item in processed_data
    # for item in processed_data:
    #     # Make prediction with self-learning model
    #     prediction = title_service.predict_title(item)
        
    #     print(f"Item - Prediction: final prediction", prediction)
        
    #     # Add prediction to item
    #     item['predicted_title'] = prediction.get('predicted_title')
    #     item['prediction_metrics'] = {
    #         'height_percent': prediction.get('height_percent'),
    #         'contrast_ratio': prediction.get('average_contrast_ratio'),
    #         'visible_ratio': prediction.get('visible_ratio')
    #     }
        
    # Run demo
    # correct_titles = {
    #     "item_1": "beauty bar with deep moisture",
    #     "item_2": "moisture",
    #     "item_3": "sensitive",
    # }
    # demo = TitlePredictorDemo()
    # results = demo.run_demo(processed_data, correct_titles)    
    # # Print results
    # print("\n==== DEMO RESULTS ====")
    # print("Initial Predictions:")
    # print(results["initial_predictions"])
    
    
    return jsonify({
        "items": processed_data,
        # "learning_stats": title_service.get_learning_stats()
    })

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
