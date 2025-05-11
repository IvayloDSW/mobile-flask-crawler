import json
import os
from models.amazon.sefl_title_predictor import SelfLearningTitlePredictor

class TitlePredictorDemo:
    def __init__(self, memory_file="title_predictor_demo.json"):
        self.memory_file = memory_file
        # Remove existing memory file for demo purposes
        if os.path.exists(memory_file):
            os.remove(memory_file)
    
    def run_demo(self, processed_data, correct_titles=None):
        """
        Run demo with your processed data
        
        Args:
            processed_data: List of data objects from AnalyzeImageTexts.analyze()
            correct_titles: Optional dict mapping item IDs to correct titles
        
        Returns:
            Results of learning process
        """
        results = {
            "initial_predictions": [],
            "learning_progress": [],
            "final_predictions": [],
            "performance": {}
        }
        
        # Step 1: Make initial predictions
        print("\n==== STEP 1: INITIAL PREDICTIONS (NO LEARNING) ====")
        for i, item in enumerate(processed_data):
            # Create an ID if not present
            if "id" not in item:
                item["id"] = f"item_{i}"
                
            predictor = SelfLearningTitlePredictor(item, self.memory_file)
            prediction = predictor.predict()
            
            print(f"Item {item['id']} - Brand: {item.get('brand', 'Unknown')}")
            print(f"  Original Title: {item.get('title', 'No title')}")
            print(f"  Predicted: {prediction.get('predicted_title', 'Failed to predict')}")
            
            # Store result
            results["initial_predictions"].append({
                "id": item["id"],
                "brand": item.get("brand", "Unknown"),
                "original_title": item.get("title", ""),
                "predicted_title": prediction.get("predicted_title", ""),
                "metrics": {
                    "height_percent": prediction.get("height_percent", 0),
                    "contrast_ratio": prediction.get("average_contrast_ratio", 0),
                    "visible_ratio": prediction.get("visible_ratio", 0)
                }
            })
            
        # Step 2: Apply feedback and learn (if correct titles provided)
        if correct_titles:
            print("\n==== STEP 2: LEARNING FROM FEEDBACK ====")
            for i, item in enumerate(processed_data[:min(5, len(processed_data))]):  # Limit to first 5 for demo
                item_id = item["id"] 
                if item_id in correct_titles:
                    # Make prediction
                    predictor = SelfLearningTitlePredictor(item, self.memory_file)
                    prediction = predictor.predict()
                    
                    # Provide correct title as feedback
                    correct_title = correct_titles[item_id]
                    predictor.give_feedback(correct_title)
                    
                    # Calculate similarity - FIX: Ensure both values are strings and handle potential None values
                    predicted_title = prediction.get("predicted_title", "")
                    if predicted_title and correct_title:  # Ensure both are not None or empty
                        similarity = predictor.calculate_similarity(predicted_title, correct_title)
                    else:
                        similarity = 0.0  # Default similarity if either value is missing
                    
                    print(f"Item {item_id} - Learning:")
                    print(f"  Predicted: {prediction.get('predicted_title', 'Failed to predict')}")
                    print(f"  Correct:   {correct_title}")
                    print(f"  Similarity: {similarity:.2f}")
                    
                    # Store progress
                    results["learning_progress"].append({
                        "id": item_id,
                        "iteration": i+1,
                        "predicted": prediction.get("predicted_title", ""),
                        "correct": correct_title,
                        "similarity": similarity
                    })
            
            # Analyze patterns after learning
            print("\n==== ANALYZING LEARNED PATTERNS ====")
            predictor = SelfLearningTitlePredictor({}, self.memory_file)
            analysis = predictor.analyze_patterns()
            
            if isinstance(analysis, dict) and "common_patterns" in analysis:
                print("Top learned patterns:")
                for pattern, count in list(analysis["common_patterns"].items())[:5]:
                    print(f"  '{pattern}': {count}")
                    
                print("\nUpdated parameters:")
                for param, value in analysis["updated_parameters"].items():
                    print(f"  {param}: {value}")
            else:
                print(f"Pattern analysis: {analysis}")
        
        # Step 3: Make final predictions with learned model
        print("\n==== STEP 3: FINAL PREDICTIONS (AFTER LEARNING) ====")
        similarities = []
        
        for item in processed_data:
            predictor = SelfLearningTitlePredictor(item, self.memory_file)
            prediction = predictor.predict()
            
            print(f"Item {item['id']} - Final Prediction:")
            print(f"  Original Title: {item.get('title', 'No title')}")
            print(f"  Predicted: {prediction.get('predicted_title', 'Failed to predict')}")
            
            # Calculate similarity if correct title is available
            if correct_titles and item["id"] in correct_titles:
                correct_title = correct_titles[item["id"]]
                predicted_title = prediction.get("predicted_title", "")
                
                # FIX: Ensure both values are strings and handle potential None values
                if predicted_title and correct_title:  # Ensure both are not None or empty
                    similarity = predictor.calculate_similarity(predicted_title, correct_title)
                else:
                    similarity = 0.0  # Default similarity if either value is missing
                
                similarities.append(similarity)
                print(f"  Correct: {correct_title}")
                print(f"  Similarity: {similarity:.2f}")
            
            # Store result
            results["final_predictions"].append({
                "id": item["id"],
                "brand": item.get("brand", "Unknown"),
                "original_title": item.get("title", ""),
                "predicted_title": prediction.get("predicted_title", ""),
                "correct_title": correct_titles.get(item["id"], "") if correct_titles else "",
                "metrics": {
                    "height_percent": prediction.get("height_percent", 0),
                    "contrast_ratio": prediction.get("average_contrast_ratio", 0),
                    "visible_ratio": prediction.get("visible_ratio", 0)
                }
            })
        
        # Calculate overall performance
        if similarities:
            results["performance"] = {
                "average_similarity": sum(similarities) / len(similarities),
                "min_similarity": min(similarities),
                "max_similarity": max(similarities),
                "items_processed": len(similarities)
            }
            
            print("\n==== PERFORMANCE SUMMARY ====")
            print(f"Items processed: {results['performance']['items_processed']}")
            print(f"Average similarity: {results['performance']['average_similarity']:.2f}")
            print(f"Min similarity: {results['performance']['min_similarity']:.2f}")
            print(f"Max similarity: {results['performance']['max_similarity']:.2f}")
        
        return results


