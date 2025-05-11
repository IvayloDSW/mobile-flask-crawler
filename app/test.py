# from flask import Flask, request, jsonify
# from self_learning_title_predictor import SelfLearningTitlePredictor
# import json
# import os

# app = Flask(__name__)

# # Assuming AnalyzeImageTexts is defined elsewhere and imported
# # from analyzer import AnalyzeImageTexts

# class TitlePredictionService:
#     def __init__(self, memory_file="title_predictor_memory.json"):
#         self.memory_file = memory_file
#         self.stats = {
#             "predictions": 0,
#             "feedback_received": 0,
#             "avg_similarity": 0.0
#         }
    
#     def predict_title(self, data):
#         """Make a prediction for a single item"""
#         predictor = SelfLearningTitlePredictor(data, self.memory_file)
#         prediction = predictor.predict()
        
#         self.stats["predictions"] += 1
#         return prediction
    
#     def batch_predict(self, items):
#         """Process multiple items and make predictions"""
#         results = []
        
#         for item in items:
#             prediction = self.predict_title(item)
            
#             # Add the prediction to results
#             results.append({
#                 "item_id": item.get("id", "unknown"),
#                 "brand": item.get("brand", ""),
#                 "original_title": item.get("title", ""),
#                 "predicted_title": prediction.get("predicted_title", ""),
#                 "confidence_metrics": {
#                     "height_percent": prediction.get("height_percent", 0),
#                     "contrast_ratio": prediction.get("average_contrast_ratio", 0),
#                     "visible_ratio": prediction.get("visible_ratio", 0)
#                 }
#             })
        
#         return results
    
#     def provide_feedback(self, item_id, original_prediction, corrected_title):
#         """Learn from feedback on a specific prediction"""
#         # Load the original data for this item (in production, you might
#         # need to store this data temporarily or retrieve it from a database)
#         data_file = f"temp_data_{item_id}.json"
        
#         if not os.path.exists(data_file):
#             return {"success": False, "error": "Original data not found"}
        
#         with open(data_file, 'r') as f:
#             data = json.load(f)
        
#         # Create predictor with original data
#         predictor = SelfLearningTitlePredictor(data, self.memory_file)
        
#         # Provide feedback for learning
#         success = predictor.give_feedback(corrected_title)
        
#         if success:
#             self.stats["feedback_received"] += 1
            
#             # Calculate similarity for stats
#             similarity = predictor.calculate_similarity(
#                 original_prediction, corrected_title)
            
#             # Update running average
#             current_total = self.stats["avg_similarity"] * (self.stats["feedback_received"] - 1)
#             new_total = current_total + similarity
#             self.stats["avg_similarity"] = new_total / self.stats["feedback_received"]
        
#         # Run pattern analysis periodically
#         if self.stats["feedback_received"] % 10 == 0:
#             predictor.analyze_patterns()
        
#         return {"success": success}
    
#     def get_learning_stats(self):
#         """Get stats about the learning progress"""
#         # Create a temporary predictor just to access memory
#         temp_predictor = SelfLearningTitlePredictor({}, self.memory_file)
        
#         top_patterns = dict(sorted(
#             temp_predictor.successful_patterns.items(), 
#             key=lambda x: x[1], 
#             reverse=True
#         )[:10])
        
#         return {
#             "predictions_made": self.stats["predictions"],
#             "feedback_received": self.stats["feedback_received"],
#             "average_similarity": round(self.stats["avg_similarity"], 2),
#             "top_learned_patterns": top_patterns,
#             "current_parameters": temp_predictor.params
#         }


# # Create a service instance
# title_service = TitlePredictionService()

# @app.route('/predict', methods=['POST'])
# def predict_titles():
#     """API endpoint for predicting titles"""
#     if not request.json or 'items' not in request.json:
#         return jsonify({"error": "Invalid request"}), 400
    
#     items = request.json['items']
    
#     # For each item, temporarily store the data for feedback purposes
#     for item in items:
#         item_id = item.get("id", f"temp_{hash(json.dumps(item))}")
#         with open(f"temp_data_{item_id}.json", 'w') as f:
#             json.dump(item, f)
    
#     # Process items and make predictions
#     results = title_service.batch_predict(items)
    
#     return jsonify({
#         "results": results,
#         "stats": title_service.get_learning_stats()
#     })

# @app.route('/feedback', methods=['POST'])
# def provide_feedback():
#     """API endpoint for providing feedback"""
#     if not request.json:
#         return jsonify({"error": "Invalid request"}), 400
    
#     item_id = request.json.get('item_id')
#     original_prediction = request.json.get('original_prediction')
#     corrected_title = request.json.get('corrected_title')
    
#     if not item_id or not corrected_title:
#         return jsonify({"error": "Missing required fields"}), 400
    
#     result = title_service.provide_feedback(item_id, original_prediction, corrected_title)
    
#     if not result.get("success"):
#         return jsonify({"error": result.get("error", "Failed to process feedback")}), 400
    
#     return jsonify({
#         "success": True,
#         "stats": title_service.get_learning_stats()
#     })

# @app.route('/learning-stats', methods=['GET'])
# def get_learning_stats():
#     """API endpoint for getting learning stats"""
#     return jsonify(title_service.get_learning_stats())


# # Example of integration with your existing Flask route:
# @app.route('/analyze-image', methods=['POST'])
# def analyze_image():
#     """Process image and predict title using both analyzers"""
#     # Get data from request
#     data = request.json
    
#     # Run existing analyzer
#     analyzerv2 = AnalyzeImageTexts(data)
#     processed_data = analyzerv2.analyze()
    
#     # Add title predictions to each item in processed_data
#     for item in processed_data:
#         # Make prediction with self-learning model
#         prediction = title_service.predict_title(item)
        
#         # Add prediction to item
#         item['predicted_title'] = prediction.get('predicted_title')
#         item['prediction_metrics'] = {
#             'height_percent': prediction.get('height_percent'),
#             'contrast_ratio': prediction.get('average_contrast_ratio'),
#             'visible_ratio': prediction.get('visible_ratio')
#         }
    
#     return jsonify({
#         "results": processed_data,
#         "learning_stats": title_service.get_learning_stats()
#     })


# # For testing purposes
# def test_predictor_with_sample():
#     """Test the predictor with sample data"""
#     # Sample data structure
#     sample_data = {
#         "id": "test123",
#         "brand": "Dove",
#         "title": "Beauty Bar Gentle Skin Cleanser Moisturizing for Gentle Soft Skin Care Original, Made With 1/4 Moisturizing Cream 3.75 oz, 14 Bars",
#         "extracted_text_rows": [
#             {
#                 "text": "14",
#                 "visible": True,
#                 "contrast_ratio": 6.6,
#                 "height": 17,
#                 "height_percentage": 7.3,
#                 "position_percentage": 3.43,
#                 "background_color_hex": "#f1f6f9", 
#                 "text_color_hex": "#4c567b"
#             },
#             {
#                 "text": "Dove PAINS BARS",
#                 "visible": True,
#                 "contrast_ratio": 5.04,
#                 "height": 40,
#                 "height_percentage": 17.17,
#                 "position_percentage": 7.3,
#                 "background_color_hex": "#f5f5f3",
#                 "text_color_hex": "#626881"
#             },
#             # ... other rows ...
#         ],
#         "image_dimensions": {
#             "height": 233,
#             "width": 304
#         }
#     }
    
#     # Make a prediction
#     prediction = title_service.predict_title(sample_data)
#     print(f"Initial prediction: {prediction['predicted_title']}")
    
#     # Provide feedback to help it learn
#     title_service.provide_feedback(
#         sample_data["id"], 
#         prediction["predicted_title"], 
#         "Dove Beauty Bar 14 Pack"
#     )
    
#     # Make another prediction to see if it learned
#     prediction2 = title_service.predict_title(sample_data)
#     print(f"After feedback: {prediction2['predicted_title']}")
    
#     return {
#         "before": prediction["predicted_title"],
#         "after": prediction2["predicted_title"],
#         "learning_stats": title_service.get_learning_stats()
#     }


# if __name__ == '__main__':
#     # Run test if in debug mode
#     test_results = test_predictor_with_sample()
#     print(f"Test results: {test_results}")
    
#     # Start Flask app
#     app.run(debug=True)