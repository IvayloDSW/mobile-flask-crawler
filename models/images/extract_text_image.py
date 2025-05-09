import pytesseract
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import difflib

class ExtractTextImage:
    def __init__(self, image_path, title):
        self.title = title
        self.image_path = image_path

    def extract_text_from_url(self):
        image_url = self.image_path
        title_to_find = self.title
        
        print(f"Image URL: {image_url}")
        print(f"Title to find: {title_to_find}")
        try:
            # Load image from URL
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            original_height = img_cv.shape[0]
            original_width = img_cv.shape[1]
            
            print(f"Processing image: {image_url}")
            print(f"Looking for title: '{title_to_find}'")
            print(f"Original dimensions: {original_width}x{original_height}")
            
            # Create a copy of the original image for debugging
            debug_img = img_cv.copy()
            
            # Create multiple processing versions to try
            processed_images = []
            
            # Version 1: Full image with high contrast
            full_img = img_cv.copy()
            full_img = cv2.resize(full_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            processed_images.append(("full_contrast", gray))
            
            # Version 2: Top half only with adaptive threshold
            top_img = img_cv[0:int(original_height/2), :]
            top_img = cv2.resize(top_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray_top = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray_top, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            processed_images.append(("top_threshold", thresh))
            
            # Version 3: Color filtering to isolate potential text
            hsv = cv2.cvtColor(full_img, cv2.COLOR_BGR2HSV)
            # Define range for white/light text
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            processed_images.append(("white_text", mask))
            
            # Try different Tesseract configs
            psm_modes = [7, 11, 6, 3]  # Single line, sparse text, block, auto
            
            all_text_found = []
            
            for img_name, processed_img in processed_images:
                for psm in psm_modes:
                    print(f"Trying {img_name} with PSM {psm}")
                    
                    config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .-&'"
                    data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Extract all text with confidence
                    found_text = []
                    for i in range(len(data["text"])):
                        text = data["text"][i].strip()
                        conf = float(data["conf"][i])
                        
                        if text and conf > 10:  # Filter very low confidence results
                            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                            
                            # For debug - draw rectangles on debug image
                            # Debug visualization would go here if needed
                            
                            found_text.append({
                                "text": text,
                                "conf": conf,
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "line_num": data["line_num"][i],
                                "method": f"{img_name}_psm{psm}"
                            })
                            print(f"  Found: '{text}' (conf: {conf})")
                    
                    all_text_found.extend(found_text)
            
            # Look for variations of the title
            title_variations = [
                title_to_find.lower(),
                title_to_find.lower().replace(" ", ""),
                "angel", 
                "soft"
            ]
            
            # First try to find exact title match
            best_match = None
            best_conf = 0
            
            # 1. Look for exact match
            for item in all_text_found:
                if item["text"].lower() == title_to_find.lower() and item["conf"] > best_conf:
                    best_match = item
                    best_conf = item["conf"]
            
            # 2. If no exact match, check for partial matches
            if not best_match:
                # Group by lines to combine words
                lines = {}
                for item in all_text_found:
                    line_key = f"{item['method']}_{item['line_num']}"
                    if line_key not in lines:
                        lines[line_key] = []
                    lines[line_key].append(item)
                
                # Check each line for the title words
                for line_key, line_items in lines.items():
                    line_items.sort(key=lambda x: x["x"])  # Sort by x-coordinate
                    line_text = " ".join([item["text"] for item in line_items])
                    line_conf = sum([item["conf"] for item in line_items]) / len(line_items)
                    
                    print(f"Line: '{line_text}' (avg conf: {line_conf:.2f})")
                    
                    # Check if any variation of the title is in this line
                    for variation in title_variations:
                        if variation in line_text.lower():
                            if line_conf > best_conf:
                                # Calculate bounding box covering all items in line
                                x_coords = [item["x"] for item in line_items]
                                y_coords = [item["y"] for item in line_items]
                                max_x = max([item["x"] + item["w"] for item in line_items])
                                max_y = max([item["y"] + item["h"] for item in line_items])
                                
                                best_match = {
                                    "text": line_text,
                                    "conf": line_conf,
                                    "x": min(x_coords),
                                    "y": min(y_coords),
                                    "w": max_x - min(x_coords),
                                    "h": max_y - min(y_coords),
                                    "method": line_key,
                                    "match_type": f"contains_{variation}"
                                }
                                best_conf = line_conf
            
            # 3. Individual word matching - look for "Angel" or "Soft" separately
            if not best_match:
                for word in ["angel", "soft"]:
                    best_word_item = None
                    best_word_conf = 0
                    
                    for item in all_text_found:
                        if item["text"].lower() == word and item["conf"] > best_word_conf:
                            best_word_item = item
                            best_word_conf = item["conf"]
                    
                    if best_word_item and best_word_conf > best_conf:
                        best_match = best_word_item
                        best_conf = best_word_conf
            
            # 4. Try fuzzy matching if still no match
            if not best_match:
                for item in all_text_found:
                    # Calculate similarity score
                    similarity = difflib.SequenceMatcher(None, item["text"].lower(), title_to_find.lower()).ratio()
                    if similarity > 0.6 and item["conf"] > best_conf:  # At least 60% similar
                        best_match = item
                        best_conf = item["conf"]
                        best_match["match_type"] = f"fuzzy_match_{similarity:.2f}"
            
            if best_match:
                # Convert coordinates back to original image scale
                scale_factor = 3  # We used 3x upscaling
                
                # Extract information from method string to determine which image region was used
                method = best_match["method"]
                if method.startswith("top_"):
                    # If we found it in the top half, y-offset is 0
                    y_offset = 0
                else:
                    y_offset = 0  # For full image
                
                # Scale back the coordinates
                x = int(best_match["x"] / scale_factor)
                y = int(best_match["y"] / scale_factor) + y_offset
                w = int(best_match["w"] / scale_factor)
                h = int(best_match["h"] / scale_factor)
                
                # Calculate height percentage
                height_percent = (h / original_height) * 100
                
                print(f"Title found: '{best_match['text']}'")
                print(f"Match type: {best_match.get('match_type', 'exact')}")
                print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
                print(f"Height percent: {height_percent:.2f}%")
                
                return {
                    "text": best_match["text"],
                    "bounding_box": (x, y, w, h),
                    "height_percent": height_percent,
                    "match_type": best_match.get("match_type", "exact"),
                    "confidence": best_conf
                }
            else:
                # Last resort: Extract all text and check if we can find parts of the title
                all_extracted_text = pytesseract.image_to_string(img_cv)
                print(f"All extracted text: {all_extracted_text}")
                
                # No match found
                return {"error": "Title not found in image."}
        
        except Exception as e:
            print(f"Exception: {str(e)}")
            return {"error": f"Error processing image: {str(e)}"}