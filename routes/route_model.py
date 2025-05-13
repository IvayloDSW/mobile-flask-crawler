from flask import Blueprint, request, jsonify
import json
from models.amazon.title_prediction_service import TitlePredictionService

# Create the blueprint
model_bp = Blueprint('model_bp', __name__)

title_service = TitlePredictionService()   

@model_bp.route('/predict', methods=['POST'])
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

@model_bp.route('/feedback', methods=['POST'])
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

@model_bp.route('/learning-stats', methods=['GET'])
def get_learning_stats():
    """API endpoint for getting learning stats"""
    return jsonify(title_service.get_learning_stats())
