from google.cloud import vision
import re
from typing import Dict, List, Any, Optional, Tuple


class AnalyzeBrandTitle:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def extract_text_from_url(self, image_url: str) -> Dict[str, Any]:
        """
        Extract text from an image URL using Google Cloud Vision API.
        Returns detailed text information including position and size.
        """
        # Initialize the Google Cloud Vision client
        client = vision.ImageAnnotatorClient()
        
        # Load the image from the URL
        image = vision.Image()
        image.source.image_uri = image_url
        
        # Perform text detection on the image
        response = client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"{response.error.message}")
        
        # Get image dimensions from the first text annotation if available
        image_height = 0
        image_width = 0
        if response.text_annotations:
            # The first annotation contains the entire text and image dimensions
            full_text = response.text_annotations[0].description
            
            # Get image dimensions from the vertices of the first annotation
            vertices = response.text_annotations[0].bounding_poly.vertices
            max_x = max(vertex.x for vertex in vertices)
            max_y = max(vertex.y for vertex in vertices)
            image_width = max_x
            image_height = max_y
        else:
            full_text = ''
        
        # Process each individual text annotation (skip the first which is the entire image text)
        text_blocks = []
        for text_annotation in response.text_annotations[1:]:
            text = text_annotation.description
            
            # Calculate bounding box coordinates
            vertices = text_annotation.bounding_poly.vertices
            top_left = vertices[0]
            bottom_right = vertices[2]
            
            # Calculate text dimensions
            width = bottom_right.x - top_left.x
            height = bottom_right.y - top_left.y
            
            # Calculate position as percentage of image dimensions
            height_percentage = (height / image_height * 100) if image_height else 0
            
            text_blocks.append({
                'text': text,
                'x': top_left.x,
                'y': top_left.y,
                'width': width,
                'height': height,
                'height_percentage': height_percentage,
                'top_left': (top_left.x, top_left.y),
                'bottom_right': (bottom_right.x, bottom_right.y)
            })
        
        # Sort text blocks by Y-coordinate (top to bottom) and then X-coordinate (left to right)
        text_blocks.sort(key=lambda block: (block['y'], block['x']))
        
        return {
            'full_text': full_text,
            'text_blocks': text_blocks,
            'image_dimensions': {
                'width': image_width,
                'height': image_height
            }
        }
    
    def identify_brand(self, text_data: Dict[str, Any], current_brand: Optional[str]) -> Tuple[str, float]:
        """
        Identify brand from text data, especially when brand is missing in the product data.
        Returns the identified brand and its height percentage.
        """
        # Common brands to check for (can be expanded)
        common_brands = ['Dove', 'Nivea', 'Olay', 'Neutrogena', 'Aveeno', 'Cetaphil', 'CeraVe', 'Dial', 'Irish Spring', 'Ivory']
        
        # If brand already exists in data, check if it's present in the image
        if current_brand:
            for block in text_data['text_blocks']:
                if current_brand.lower() in block['text'].lower():
                    return current_brand, block['height_percentage']
        
        # Look for common brands in the text blocks
        for block in text_data['text_blocks']:
            for brand in common_brands:
                if brand.lower() in block['text'].lower():
                    # Check if this is likely a brand by looking at position and size
                    # Brands are typically at the top with larger text
                    if block['height_percentage'] > 5:  # Assuming brands are at least 5% of image height
                        return brand, block['height_percentage']
        
        # Look for standalone words that might be brands (larger text, typically at the top)
        for block in text_data['text_blocks']:
            # Check if text is a single word that could be a brand (not a number or common descriptor)
            if (block['height_percentage'] > 5 and 
                len(block['text'].split()) == 1 and 
                not re.match(r'^\d+\.?\d*$', block['text']) and
                block['text'].lower() not in ['oz', 'ml', 'original', 'new', 'pack']):
                return block['text'], block['height_percentage']
        
        return current_brand or "Unknown", 0
    
    def identify_title_components(self, text_data: Dict[str, Any], product_title: str) -> Dict[str, Any]:
        """
        Extract components from the image that match parts of the product title.
        Handles multi-row text while maintaining title word sequence.
        Returns matching text blocks and their height percentages.
        """
        # Break the product title into words and normalize
        title_words = [word.lower().strip() for word in product_title.split()]
        
        # Sort text blocks by vertical position (top-to-bottom, then left-to-right)
        sorted_blocks = sorted(text_data['text_blocks'], key=lambda b: (b['y'], b['x']))
        
        # Group text blocks by rows based on vertical position (y-coordinate)
        rows = []
        current_row = []
        current_y = None
        
        # Y-coordinate threshold for considering blocks on the same row (adjust if needed)
        threshold = max(text_data['image_dimensions']['height'] * 0.03, 10)  # 3% of image height or 10px
        
        for block in sorted_blocks:
            if current_y is None:
                current_y = block['y']
                current_row.append(block)
            elif abs(block['y'] - current_y) <= threshold:
                # Same row
                current_row.append(block)
            else:
                # New row
                if current_row:
                    # Sort blocks in row by x-coordinate (left-to-right)
                    current_row.sort(key=lambda b: b['x'])
                    rows.append(current_row)
                current_row = [block]
                current_y = block['y']
                
        # Add the last row if not empty
        if current_row:
            current_row.sort(key=lambda b: b['x'])
            rows.append(current_row)
        
        # Process rows to find sequential matches from the title
        title_components = []
        matched_indices = set()  # Track which title words have been matched
        
        # Process each row of text
        for row_idx, row in enumerate(rows):
            row_text = " ".join([block['text'].lower() for block in row])
            row_blocks = row
            
            # Try to find consecutive title words in this row
            for start_idx in range(len(title_words)):
                # Skip if already matched
                if start_idx in matched_indices:
                    continue
                    
                # Try increasing sequences of words
                best_match_length = 0
                best_match_text = ""
                
                for length in range(1, len(title_words) - start_idx + 1):
                    end_idx = start_idx + length
                    # Skip if any word in range already matched
                    if any(i in matched_indices for i in range(start_idx, end_idx)):
                        continue
                        
                    title_phrase = " ".join(title_words[start_idx:end_idx])
                    
                    # Check if this sequence is in the row text
                    if title_phrase in row_text:
                        if length > best_match_length:
                            best_match_length = length
                            best_match_text = title_phrase
                
                # If found a match, record it
                if best_match_length > 0:
                    # Mark these words as matched
                    for i in range(start_idx, start_idx + best_match_length):
                        matched_indices.add(i)
                        
                    # Calculate height metrics for this row
                    row_heights = [block['height'] for block in row_blocks]
                    avg_height = sum(row_heights) / len(row_heights) if row_heights else 0
                    height_percentage = avg_height / text_data['image_dimensions']['height'] * 100 if text_data['image_dimensions']['height'] else 0
                    
                    # Calculate combined position
                    left_x = min(block['x'] for block in row_blocks)
                    top_y = min(block['y'] for block in row_blocks)
                    right_x = max(block['x'] + block['width'] for block in row_blocks)
                    bottom_y = max(block['y'] + block['height'] for block in row_blocks)
                    
                    title_components.append({
                        'text': " ".join([block['text'] for block in row_blocks]),
                        'matched_phrase': best_match_text,
                        'height_percentage': height_percentage,
                        'row_index': row_idx,
                        'title_word_indices': list(range(start_idx, start_idx + best_match_length)),
                        'position': ((left_x, top_y), (right_x, bottom_y))
                    })
        
        # Process single important words if not already matched
        important_keywords = ['original', 'moisture', 'beauty', 'deep', 'sensitive', 'cleanser', 'soft', 'skin']
        for row_idx, row in enumerate(rows):
            for block in row:
                block_text = block['text'].lower()
                
                for keyword in important_keywords:
                    # Find the keyword position in title
                    for i, word in enumerate(title_words):
                        if keyword == word and i not in matched_indices and keyword in block_text:
                            matched_indices.add(i)
                            
                            title_components.append({
                                'text': block['text'],
                                'matched_phrase': keyword,
                                'height_percentage': block['height_percentage'],
                                'row_index': row_idx,
                                'title_word_indices': [i],
                                'position': (block['top_left'], block['bottom_right'])
                            })
        
        # Sort components by their order in the title
        title_components.sort(key=lambda c: min(c['title_word_indices']) if c['title_word_indices'] else float('inf'))
        
        # Find sequential rows that match title sequence
        sequential_components = []
        last_row_idx = -1
        last_title_idx = -1
        
        for component in title_components:
            min_title_idx = min(component['title_word_indices'])
            
            # Check if this component maintains sequence
            if min_title_idx > last_title_idx and component['row_index'] >= last_row_idx:
                sequential_components.append(component)
                last_row_idx = component['row_index']
                last_title_idx = max(component['title_word_indices'])
        
        return {
            'components': sequential_components,
            'avg_height_percentage': sum(c['height_percentage'] for c in sequential_components) / len(sequential_components) if sequential_components else 0,
            'all_matches': title_components,  # Include all matches for debugging
            'title_coverage': len(matched_indices) / len(title_words) if title_words else 0
        }

    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze each product to extract and verify brand and title information.
        """
        processed_data = []
        
        for item in self.data:
            if not isinstance(item, dict):
                print("Item is not a dictionary.")
                processed_data.append(item)
                continue
                
            # Get basic product info
            title = item.get('title', '')
            current_brand = item.get('brand')
            image_url = item.get('image', '')
            
            result_item = item.copy()
            
            if image_url:
                try:
                    # Extract text from the image
                    text_data = self.extract_text_from_url(image_url)
                    
                    # Save full detected text
                    result_item['detected_text'] = text_data['full_text']
                    
                    # Identify/verify brand
                    identified_brand, brand_height_pct = self.identify_brand(text_data, current_brand)
                    
                    # If brand was null or "Unknown", update it
                    if not current_brand or current_brand == "Unknown":
                        result_item['brand'] = identified_brand
                    
                    result_item['brand_details'] = {
                        'identified_brand': identified_brand,
                        'height_percentage': brand_height_pct
                    }
                    
                    # Identify title components
                    title_analysis = self.identify_title_components(text_data, title)
                    
                    # Enhanced title details with sequence information
                    result_item['title_details'] = {
                        'components': title_analysis['components'],
                        'avg_height_percentage': title_analysis['avg_height_percentage'],
                        'title_coverage': title_analysis['title_coverage']
                    }
                    
                    # Build a sequenced title from components
                    if title_analysis['components']:
                        sequenced_title_parts = []
                        for component in title_analysis['components']:
                            sequenced_title_parts.append({
                                'text': component['text'],
                                'matched_phrase': component['matched_phrase'],
                                'height_percentage': component['height_percentage'],
                                'row_index': component['row_index']
                            })
                        
                        result_item['sequenced_title'] = sequenced_title_parts
                    
                    # Print summary of findings
                    print(f"Product: {title[:50]}...")
                    print(f"Brand: {identified_brand} (Height: {brand_height_pct:.2f}%)")
                    print(f"Found {len(title_analysis['components'])} sequential title components")
                    print(f"Title coverage: {title_analysis['title_coverage']*100:.1f}%")
                    
                    # Print detected title components in sequence
                    if title_analysis['components']:
                        print("Sequential title components:")
                        for i, comp in enumerate(title_analysis['components']):
                            print(f"  {i+1}. '{comp['text']}' (matched: '{comp['matched_phrase']}', height: {comp['height_percentage']:.1f}%)")
                    print("---")
                    
                except Exception as e:
                    print(f"Error processing image {image_url}: {str(e)}")
                    result_item['processing_error'] = str(e)
            
            processed_data.append(result_item)
        
        return processed_data
    
    def get_data(self):
        """
        Return the processed data.
        """
        return self.data


# Example usage in Flask:
"""
@app.route('/analyze_products', methods=['POST'])
def analyze_products():
    data = request.json.get('products', [])
    
    analyzer = AnalyzeBrandTitle(data)
    processed_data = analyzer.analyze()
    
    return jsonify({
        'processed_products': processed_data
    })
"""