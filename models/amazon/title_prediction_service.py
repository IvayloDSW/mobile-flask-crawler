
from models.amazon.sefl_title_predictor import SelfLearningTitlePredictor

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