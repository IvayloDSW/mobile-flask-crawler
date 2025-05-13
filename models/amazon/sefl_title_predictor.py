import re
import json
import os
from collections import defaultdict
from datetime import datetime


default_params = {
    "min_word_length": 3,
    "visibility_weight": 1.5,
    "contrast_weight": 1.2,
    "height_weight": 1.0,
    "word_match_weight": 2.0,
    "position_penalty": 0.05
}

class SelfLearningTitlePredictor:
    def __init__(self, data, memory_file="title_predictor_memory.json"):
        # Original data
        self.data = data
        self.item = data or {}
        self.memory_file = memory_file
        
        # Initialize previous predictions
        self.previous_predictions = []  # Changed from {} to [] since it's treated as a list later
        
        # Load any existing learning parameters
        memory = self.load_memory()
        self.parameters = memory.get('parameters', {})
        
        # Set defaults if parameters don't exist yet
        if not self.parameters:
            self.parameters = {
                'text_height_weight': 0.6,
                'contrast_weight': 0.3,
                'visibility_weight': 0.1,
                # Other default parameters
            }
        self.brand = (data.get("brand") or "").lower()
        full_title = data.get("title", "").lower()
        self.title = full_title.split(',')[0].strip()
        self.rows = data.get("extracted_text_rows", [])
        self.image_height = data.get("image_dimensions", {}).get("height", 1)
        self.image_width = data.get("image_dimensions", {}).get("width", 1)
        self.filtered_rows = []
        
        # Memory management
        self.memory = memory
        
        # Learning parameters (will be adjusted through feedback)
        self.params = self.memory.get("params", {})
        for key, val in default_params.items():
            self.params.setdefault(key, val)
        
        # Track performance
        self.previous_predictions = self.memory.get("previous_predictions", [])
        self.successful_patterns = defaultdict(int, self.memory.get("successful_patterns", {}))

        self.brand_specific_rules = self.memory.get("brand_rules", {}).get(self.brand, {})

    def load_memory(self):
        """Load learning memory from file if available"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                return {"params": {}, "previous_predictions": [], 
                        "successful_patterns": {}, "brand_rules": {}}
        return {"params": {}, "previous_predictions": [], 
                "successful_patterns": {}, "brand_rules": {}}
    
    def save_memory(self):
        """Save learning memory to file"""
        # Make sure we don't exceed list bounds - use integer indexing rather than slices
        predictions_to_save = self.previous_predictions
        if len(predictions_to_save) > 100:
            predictions_to_save = predictions_to_save[-100:]
            
        # Convert defaultdict to regular dict for JSON serialization
        memory_to_save = {
            "params": self.params,
            "previous_predictions": predictions_to_save,  # Fixed potential slice issue
            "successful_patterns": dict(self.successful_patterns),
            "brand_rules": self.memory.get("brand_rules", {})
        }
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory_to_save, f)
    
    def filter_rows(self):
        """Filter text rows using learned parameters and rules"""
        def is_valid(row):
            text = row["text"].strip().lower()
            words = text.split()
            
            # Apply brand-specific rules if available
            if self.brand in self.memory.get("brand_rules", {}):
                brand_rules = self.memory["brand_rules"][self.brand]
                
                # Check for any skip patterns
                if "skip_patterns" in brand_rules:
                    for pattern in brand_rules["skip_patterns"]:
                        if re.search(pattern, text):
                            return False
            
            # Rule 1: Exclude exact match with brand
            if text == self.brand:
                return False
                
            # Rule 2: Exclude single numbers unless learned to be important
            if re.fullmatch(r'\d+', text) and not self.is_learned_important_number(text):
                return False
                
            # Rule 3: Exclude short single-word texts (using learned parameter)
            if len(words) == 1 and words[0] and len(words[0]) < self.params["min_word_length"]:
                return False
                
            # Rule 4: Low contrast text is less likely to be important
            if row["contrast_ratio"] < 2.0 and not row["visible"]:
                return False
                
            return True
        
        self.filtered_rows = [row for row in self.rows if is_valid(row)]
    
    def is_learned_important_number(self, text):
        """Check if a numeric value has been learned as important"""
        # For product counts, weights, etc.
        important_pattern = self.successful_patterns.get(f"number_{text}", 0)
        return important_pattern > 2  # If we've learned this number is important multiple times
    
    def score_row(self, row):
        """Score a text row based on learned parameters"""
        text = row["text"].lower()
        words = set(text.split())
        title_words = set(self.title.split())
        
        # Calculate base score
        score = 0
        
        # Factor 1: Visibility
        if row["visible"]:
            score += self.params["visibility_weight"]
        
        # Factor 2: Contrast ratio (better contrast = more important)
        if row["contrast_ratio"] > 4.5:  # WCAG AA standard for normal text
            score += self.params["contrast_weight"]
        
        # Factor 3: Height percentage (bigger = more important)
        score += (row["height_percentage"] / 10) * self.params["height_weight"]
        
        # Factor 4: Word match with known title
        match_count = len(words & title_words)
        score += match_count * self.params["word_match_weight"]
        
        # Factor 5: Position penalty (lower in image = less likely to be title)
        position_penalty = (row["position_percentage"] / 100) * self.params["position_penalty"]
        score -= position_penalty
        
        # Factor 6: Previously successful patterns
        for word in words:
            pattern_score = self.successful_patterns.get(word, 0) * 0.5
            score += pattern_score
        
        # Factor 7: Brand-specific boosting
        if self.brand_specific_rules and "boost_terms" in self.brand_specific_rules:
            for term in self.brand_specific_rules["boost_terms"]:
                if term in text:
                    score += self.brand_specific_rules.get("boost_value", 1.0)
        
        return score
    
    def get_best_rows(self):
        """Get top rows by score"""
        if not self.filtered_rows:
            return []
            
        # Score all rows
        scored_rows = [(row, self.score_row(row)) for row in self.filtered_rows]
        
        # Sort by score (descending)
        scored_rows.sort(key=lambda x: x[1], reverse=True)
        
        # Choose top 1-3 rows depending on score distribution
        top_score = scored_rows[0][1]
        selected_rows = [row for row, score in scored_rows if score >= top_score * 0.7]
        
        # Limit to top 3 rows maximum
        return selected_rows[:3]
    
    def build_title_and_metrics(self, selected_rows):
        """Build title from selected rows and calculate metrics"""
        # Sort rows by vertical position
        selected_rows.sort(key=lambda x: x["position_percentage"])
        
        combined_text = " ".join(row["text"] for row in selected_rows)
        total_height = sum(row["height"] for row in selected_rows)
        total_height_pct = sum(row["height_percentage"] for row in selected_rows)
        avg_contrast = sum(row["contrast_ratio"] for row in selected_rows) / len(selected_rows)
        visible_rows = [row for row in selected_rows if row["visible"]]
        
        return {
            "predicted_title": combined_text,
            "total_height": total_height,
            "height_percent": round(total_height_pct, 2),
            "average_contrast_ratio": round(avg_contrast, 2),
            "visible_ratio": round(len(visible_rows) / len(selected_rows), 2),
            "selected_rows": [row["text"] for row in selected_rows]
        }
    
    def predict(self):
        """Make a prediction based on current learning state"""
        self.filter_rows()
        
        if not self.filtered_rows:
            return {"error": "No suitable rows found"}
        
        selected_rows = self.get_best_rows()
        prediction = self.build_title_and_metrics(selected_rows)
        
        print(f"Prediction: {prediction}")
        
        # Store this prediction for learning
        # Option 1: If self.previous_predictions should be a list
        # Initialize self.previous_predictions as a list if it doesn't exist or is a dict
        if not hasattr(self, 'previous_predictions') or not isinstance(self.previous_predictions, list):
            self.previous_predictions = []

        # Now append to the list
        self.previous_predictions.append({
            "input": {
                "brand": self.brand,
                "original_title": self.title
            },
            "prediction": prediction,
            "parameters": self.params.copy(),
            "feedback": None
        })
        
        return prediction
    
    def give_feedback(self, corrected_title):
        """Learn from feedback for future predictions"""
        # Handle different types of previous_predictions
        if isinstance(self.previous_predictions, list) and len(self.previous_predictions) > 0:
            latest_prediction = self.previous_predictions[-1]
            if isinstance(latest_prediction, dict) and "prediction" in latest_prediction:
                predicted_title = latest_prediction["prediction"].get("predicted_title", "")
            else:
                # Make a new prediction
                prediction_result = self.predict()
                predicted_title = prediction_result.get("predicted_title", "")
        elif isinstance(self.previous_predictions, dict):
            # Handle the case where previous_predictions is a dict
            predicted_title = self.previous_predictions.get("predicted_title", "")
        else:
            # Make a prediction to populate previous_predictions
            prediction_result = self.predict()
            predicted_title = prediction_result.get("predicted_title", "")
        
        # Calculate similarity between prediction and corrected title
        similarity = self.calculate_similarity(predicted_title, corrected_title)
        
        # Get item data - ensure it's available
        item_data = getattr(self, "item", {})
        item_id = item_data.get("id", "unknown") if isinstance(item_data, dict) else "unknown"
        
        # Save the feedback for learning
        feedback_item = {
            "item_data": item_data,
            "predicted": predicted_title,
            "corrected": corrected_title,
            "similarity": similarity,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update memory with feedback
        memory = self.load_memory()
        memory["feedback_items"] = memory.get("feedback_items", [])
        memory["feedback_items"].append(feedback_item)
        
        # Update learning patterns based on feedback
        self._update_learning_patterns(memory, feedback_item)
        
        # Save updated memory
        self.save_memory()
        
        return {
            "success": True,
            "similarity": similarity,
            "item_id": item_id
        }
    
    def _update_learning_patterns(self, memory, feedback_item):
        """Update learning patterns based on feedback"""
        predicted = feedback_item.get("predicted", "").lower()
        corrected = feedback_item.get("corrected", "").lower()
        
        predicted_words = set(predicted.split())
        corrected_words = set(corrected.split())
        
        # Find common words that appear in both predicted and corrected
        common_words = predicted_words & corrected_words
        
        # Update pattern weights for common words
        for word in common_words:
            if len(word) >= self.params["min_word_length"]:
                self.successful_patterns[word] = self.successful_patterns.get(word, 0) + 1
        
        # Learn from words in corrected that weren't predicted
        for word in corrected_words - predicted_words:
            if len(word) >= self.params["min_word_length"]:
                self.successful_patterns[word] = self.successful_patterns.get(word, 0) + 0.5
    
    def analyze_patterns(self):
        """Analyze patterns and improve prediction model"""
        if len(self.previous_predictions) < 5:
            return "Not enough data for pattern analysis"
        
        successful_preds = []
        for p in self.previous_predictions:
            if isinstance(p, dict) and p.get("feedback") and "prediction" in p:
                similarity = self.calculate_similarity(
                    p["prediction"].get("predicted_title", ""), 
                    p.get("feedback", "")
                )
                if similarity > 0.7:
                    successful_preds.append(p)
        
        if not successful_preds:
            return "No successful predictions for analysis"
        
        # Extract common patterns
        common_patterns = defaultdict(int)
        
        for pred in successful_preds:
            pred_words = pred["prediction"]["predicted_title"].lower().split()
            feedback_words = pred.get("feedback", "").lower().split()
            
            # Find words that consistently appear in both
            common = set(pred_words) & set(feedback_words)
            for word in common:
                common_patterns[word] += 1
        
        # Update successful patterns
        for word, count in common_patterns.items():
            if count >= 2:  # If pattern appears in multiple successful predictions
                self.successful_patterns[word] = max(self.successful_patterns.get(word, 0), count)
        
        # Optimize parameters based on successful predictions
        if len(successful_preds) >= 3:
            avg_visibility = 0
            avg_height = 0
            
            for p in successful_preds:
                if "prediction" in p:
                    avg_visibility += p["prediction"].get("visible_ratio", 0)
                    avg_height += p["prediction"].get("height_percent", 0)
            
            avg_visibility /= len(successful_preds)
            avg_height /= len(successful_preds)
            
            # Adjust weights based on averages
            if avg_visibility > 0.7:
                self.params["visibility_weight"] *= 1.05
            if avg_height > 12:
                self.params["height_weight"] *= 1.05
        
        self.save_memory()
        return {
            "analyzed_predictions": len(successful_preds),
            "common_patterns": dict(sorted(common_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "updated_parameters": self.params
        }
    
    def calculate_similarity(self, text1, text2):
        """Calculate word-based similarity between two texts"""
        if not text1 or not text2:
            return 0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        return len(words1 & words2) / max(1, len(words1 | words2))
    
    def simulate_learning(self, training_data):
        """Simulate learning process with multiple examples"""
        results = []
        
        for item in training_data:
            # Set current data
            self.data = item["data"]
            self.brand = item["data"].get("brand", "").lower()
            self.title = item["data"].get("title", "").lower()
            self.rows = item["data"].get("extracted_text_rows", [])

            # Make prediction
            prediction = self.predict()

            if "error" in prediction:
                results.append({
                    "original_title": self.title,
                    "predicted": None,
                    "correct": item["correct_title"],
                    "similarity": 0,
                    "error": prediction["error"]
                })
                continue  # Skip feedback for failed predictions

            # Provide feedback
            self.give_feedback(item["correct_title"])

            results.append({
                "original_title": self.title,
                "predicted": prediction["predicted_title"],
                "correct": item["correct_title"],
                "similarity": self.calculate_similarity(prediction["predicted_title"], item["correct_title"])
            })

        # Analyze patterns after training
        self.analyze_patterns()

        return {
            "results": results,
            "learned_patterns": dict(sorted(self.successful_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "updated_params": self.params
        }



# Example usage:
if __name__ == "__main__":
    # Example data structure
    data = {
        "brand": "Dove",
        "title": "Beauty Bar Gentle Skin Cleanser Moisturizing for Gentle Soft Skin Care Original Made With 1/4 Moisturizing Cream 3.75 oz, 14 Bars",
        "extracted_text_rows": [
            # ... text rows from image ...
        ],
        "image_dimensions": {
            "height": 233,
            "width": 304
        }
    }
    
    # Create predictor
    predictor = SelfLearningTitlePredictor(data)
    
    # Make prediction
    prediction = predictor.predict()
    print(f"Predicted: {prediction['predicted_title']}")
    
    # Provide feedback for learning
    predictor.give_feedback("Dove Beauty Bar Original 14 Pack")
    
    # Make another prediction to see if learning improved results
    prediction2 = predictor.predict()
    print(f"New prediction: {prediction2['predicted_title']}")
    
    # Analyze patterns
    analysis = predictor.analyze_patterns()
    print(f"Analysis: {analysis}")