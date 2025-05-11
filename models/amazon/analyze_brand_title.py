from google.cloud import vision
import re
from typing import Dict, List, Any, Optional, Tuple
import math


class ProductTitleExtractor:
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
            area = width * height
            
            # Calculate position as percentage of image dimensions
            height_percentage = (height / image_height * 100) if image_height else 0
            width_percentage = (width / image_width * 100) if image_width else 0
            
            text_blocks.append({
                'text': text,
                'x': top_left.x,
                'y': top_left.y,
                'width': width,
                'height': height,
                'area': area,
                'height_percentage': height_percentage,
                'width_percentage': width_percentage,
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
        # If brand already exists in data, check if it's present in the image
        if current_brand:
            for block in text_data['text_blocks']:
                if current_brand.lower() in block['text'].lower():
                    return current_brand, block['height_percentage']
        
        # Look for standalone words that might be brands (larger text, typically at the top)
        # Sort by height percentage (largest first) and prioritize upper portions of image
        brand_candidates = []
        
        for block in text_data['text_blocks']:
            # Consider only blocks in the top 40% of the image
            relative_y_pos = block['y'] / text_data['image_dimensions']['height']
            
            # Consider only larger text elements with a single word
            if (block['height_percentage'] > 5 and 
                len(block['text'].split()) == 1 and 
                not re.match(r'^\d+\.?\d*$', block['text']) and
                relative_y_pos < 0.4 and
                len(block['text']) > 1):  # Avoid single characters
                
                brand_candidates.append((block['text'], block['height_percentage'], block['y']))
        
        # Sort by size (larger first), then by position (top first)
        brand_candidates.sort(key=lambda x: (-x[1], x[2]))
        
        # Return the most likely brand candidate
        if brand_candidates:
            return brand_candidates[0][0], brand_candidates[0][1]
        
        return current_brand or "Unknown", 0
    
    def identify_visual_containers(self, text_blocks: List[Dict[str, Any]], image_dims: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Identify visual containers of text - groups of text blocks that seem to belong together.
        Uses both proximity and visual alignment to group text.
        """
        containers = []
        processed_blocks = set()
        
        # Sort blocks by size (largest first) to prioritize prominent text
        sorted_blocks = sorted(text_blocks, key=lambda b: -b['area'])
        
        # Define proximity thresholds based on image dimensions
        # These thresholds determine when blocks are considered part of the same container
        h_threshold = image_dims['width'] * 0.05  # 5% of image width
        v_threshold = image_dims['height'] * 0.03  # 3% of image height
        
        for i, block in enumerate(text_blocks):
            if i in processed_blocks:
                continue
                
            # Start a new container with this block
            container = {
                'blocks': [block],
                'top': block['y'],
                'left': block['x'],
                'right': block['x'] + block['width'],
                'bottom': block['y'] + block['height'],
                'area': block['area'],
                'text_content': [block['text']]
            }
            processed_blocks.add(i)
            
            # Find related blocks for this container
            changed = True
            while changed:
                changed = False
                for j, other_block in enumerate(text_blocks):
                    if j in processed_blocks:
                        continue
                        
                    # Check if blocks are in close proximity or aligned
                    horizontally_close = (
                        (other_block['x'] >= container['left'] - h_threshold and 
                         other_block['x'] <= container['right'] + h_threshold) or
                        (other_block['x'] + other_block['width'] >= container['left'] - h_threshold and 
                         other_block['x'] + other_block['width'] <= container['right'] + h_threshold)
                    )
                    
                    vertically_close = (
                        (other_block['y'] >= container['top'] - v_threshold and 
                         other_block['y'] <= container['bottom'] + v_threshold) or
                        (other_block['y'] + other_block['height'] >= container['top'] - v_threshold and 
                         other_block['y'] + other_block['height'] <= container['bottom'] + v_threshold)
                    )
                    
                    # Check for alignment (blocks that line up horizontally or vertically)
                    horizontally_aligned = (
                        abs(other_block['y'] - container['top']) < v_threshold or
                        abs(other_block['y'] + other_block['height'] - container['bottom']) < v_threshold
                    )
                    
                    vertically_aligned = (
                        abs(other_block['x'] - container['left']) < h_threshold or
                        abs(other_block['x'] + other_block['width'] - container['right']) < h_threshold
                    )
                    
                    if (horizontally_close and vertically_close) or (horizontally_aligned and vertically_close) or (vertically_aligned and horizontally_close):
                        # Add this block to the container
                        container['blocks'].append(other_block)
                        container['text_content'].append(other_block['text'])
                        container['top'] = min(container['top'], other_block['y'])
                        container['left'] = min(container['left'], other_block['x'])
                        container['right'] = max(container['right'], other_block['x'] + other_block['width'])
                        container['bottom'] = max(container['bottom'], other_block['y'] + other_block['height'])
                        container['area'] += other_block['area']
                        processed_blocks.add(j)
                        changed = True
            
            # Calculate container properties
            container['width'] = container['right'] - container['left']
            container['height'] = container['bottom'] - container['top']
            container['block_count'] = len(container['blocks'])
            
            # Sort blocks within container by vertical position
            container['blocks'].sort(key=lambda b: (b['y'], b['x']))
            
            # Get combined text (preserving order)
            container['combined_text'] = ' '.join(block['text'] for block in container['blocks'])
            
            # Calculate average text size in the container
            if container['blocks']:
                container['avg_height_percentage'] = sum(b['height_percentage'] for b in container['blocks']) / len(container['blocks'])
            else:
                container['avg_height_percentage'] = 0
                
            containers.append(container)
        
        # Sort containers by position (top to bottom)
        containers.sort(key=lambda c: c['top'])
        
        return containers
    
    def extract_product_attributes(self, 
                                  text_data: Dict[str, Any], 
                                  current_title: str,
                                  identified_brand: str) -> Dict[str, Any]:
        """
        Extract key product attributes from the image text.
        Focuses on specific packaging patterns for soap products.
        Specifically targets text from colored containers (often blue) on soap packaging.
        """
        # Get image dimensions
        image_dims = text_data['image_dimensions']
        
        # Identify visual containers of text
        containers = self.identify_visual_containers(text_data['text_blocks'], image_dims)
        
        # Prepare clean version of title for matching - IMPORTANT: Remove the brand from title for matching
        clean_title = re.sub(r'[,.\'"-]', ' ', current_title).lower()
        
        # Remove brand name from title before splitting into words
        if identified_brand and identified_brand != "Unknown":
            clean_title = clean_title.replace(identified_brand.lower(), '')
        
        # Split into words and filter out empty strings
        title_words = [word for word in clean_title.split() if word]
        
        # 1. First identify key areas by relative position
        # For Dove soap, the brand is typically at the top, variant in the middle,
        # quantity in upper-right, and the key title text often in a colored band
        
        # Extract product type/variant (usually medium to large text below brand)
        product_variant = None
        variant_score = 0
        
        # Identify product variants (looking at medium-large text below brand, excluding numbers)
        for container in containers:
            # Skip very small text or likely to be brand/quantity
            if container['avg_height_percentage'] < 3:
                continue
                
            if identified_brand and identified_brand.lower() in container['combined_text'].lower():
                continue
                
            # Skip containers with numbers and units (likely quantity info)
            if re.search(r'\d+\s*(bars?|pains?|pack|oz|g|ml)', container['combined_text'], re.IGNORECASE):
                continue
                
            # Check for common product descriptors in soap products
            variant_keywords = ['original', 'sensitive', 'moisture', 'moisturizing', 'gentle', 
                             'beauty', 'deep clean', 'deep', 'clean', 'daily', 'soft']
            
            container_text = container['combined_text'].lower()
            
            # Single words are more likely to be variants
            is_single_word = len(container_text.split()) <= 2
            
            # Calculate a score based on text size and keyword matches
            score = container['avg_height_percentage']
            
            # Give extra points for single words that are variant keywords
            if is_single_word:
                score += 5
                for keyword in variant_keywords:
                    if keyword == container_text:
                        score += 10
            
            # Also check for partial matches
            for keyword in variant_keywords:
                if keyword in container_text:
                    score += 5
            
            # Prioritize if words from the title appear
            title_word_matches = sum(1 for word in title_words if len(word) > 3 and word in container_text.split())
            score += title_word_matches * 2
            
            # Update if this is the best variant candidate so far
            if score > variant_score:
                product_variant = container['combined_text']
                variant_score = score
        
        # Extract quantity information (e.g., "14 BARS")
        quantity_info = None
        quantity_confidence = 0
        
        for container in containers:
            container_text = container['combined_text']
            
            # Look for number patterns followed by units
            quantity_patterns = [
                r'(\d+)\s*(bars?|pains?|pack|ct|count)',
                r'(\d+[\.\d]*)\s*(oz|ounce|g|grams?|ml)',
                r'(\d+)\s*x\s*(\d+[\.\d]*)\s*(oz|g|ml)'
            ]
            
            for pattern in quantity_patterns:
                matches = re.search(pattern, container_text, re.IGNORECASE)
                if matches:
                    # Calculate confidence score based on text characteristics
                    confidence = container['avg_height_percentage']
                    
                    # Boost confidence if the text appears in upper right (common for quantity)
                    rel_x = container['left'] / image_dims['width']
                    rel_y = container['top'] / image_dims['height']
                    if rel_x > 0.6 and rel_y < 0.4:
                        confidence += 15
                        
                    # If this is better than what we've found so far
                    if confidence > quantity_confidence:
                        quantity_info = container_text
                        quantity_confidence = confidence
        
        # Find the blue container that typically contains main descriptive text
        # This is especially common in Dove soap packaging
        color_band_text = None
        color_band_score = 0
        
        for container in containers:
            # Skip very small text
            if container['avg_height_percentage'] < 2:
                continue
                
            # Skip containers likely to be brand or variant
            if identified_brand and identified_brand.lower() in container['combined_text'].lower():
                continue
                
            if product_variant and container['combined_text'] == product_variant:
                continue
                
            if quantity_info and container['combined_text'] == quantity_info:
                continue
            
            # Key characteristics of the colored band text:
            # 1. Multi-word phrases (typically 3+ words)
            # 2. Often contains keywords like "beauty bar", "moisture", etc.
            # 3. Width tends to be significant (wider than taller)
            # 4. Often positioned in the middle section of the package
            
            word_count = len(container['combined_text'].split())
            width_to_height_ratio = container['width'] / container['height'] if container['height'] > 0 else 0
            
            # Calculate relative position
            rel_y = container['top'] / image_dims['height']
            
            # Calculate score for colored band
            score = 0
            
            # Multi-word text gets higher score
            if word_count >= 3:
                score += 10
                
            # Wide container is more likely to be the color band
            if width_to_height_ratio > 3:
                score += 8
                
            # Position in middle of packaging (typical for Dove)
            if 0.25 < rel_y < 0.75:
                score += 5
                
            # Check for beauty/soap related keywords
            color_band_keywords = ['beauty bar', 'with deep moisture', 'moisturizing', 
                                'skin', 'cleanser', 'hydration', 'cream']
            
            for keyword in color_band_keywords:
                if keyword in container['combined_text'].lower():
                    score += 12
            
            # Also match against product title (without brand)
            container_text = container['combined_text'].lower()
            matched_words = sum(1 for word in title_words if len(word) > 3 and word in container_text)
            score += matched_words * 3
                
            if score > color_band_score:
                color_band_text = container['combined_text']
                color_band_score = score
        
        # Extract other product description elements (significant text that matches title words)
        descriptions = []
        
        for container in containers:
            # Skip very small text and already identified elements
            if container['avg_height_percentage'] < 3:
                continue
                
            if (identified_brand and identified_brand.lower() in container['combined_text'].lower() and
                len(container['combined_text'].split()) <= 2):
                continue
                
            if product_variant and container['combined_text'] == product_variant:
                continue
                
            if quantity_info and container['combined_text'] == quantity_info:
                continue
                
            if color_band_text and container['combined_text'] == color_band_text:
                continue
            
            # Check how many title words match this container
            container_text = container['combined_text'].lower()
            matched_words = sum(1 for word in title_words if len(word) > 3 and word in container_text)
            
            # Calculate a relevance score
            relevance = (matched_words / len(title_words)) if title_words else 0
            
            if relevance > 0.1 or container['avg_height_percentage'] > 5:
                descriptions.append({
                    'text': container['combined_text'],
                    'relevance': relevance,
                    'height_percentage': container['avg_height_percentage'],
                    'position': {'top': container['top'], 'left': container['left']}
                })
        
        # Sort descriptions by relevance
        descriptions.sort(key=lambda d: -d['relevance'])
        
        # Return all extracted attributes with the new color band text
        return {
            'brand': identified_brand,
            'variant': product_variant,
            'quantity': quantity_info,
            'color_band_text': color_band_text,
            'descriptions': descriptions,
            'containers': containers
        }
    
    def reconstruct_title_from_image(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct a product title based on extracted image attributes.
        Creates a structured version of what appears on the packaging.
        Prioritizes color band text for soap products.
        MODIFIED: Now excludes brand from reconstructed title.
        """
        title_components = []
        confidence = 0
        
        # We're specifically NOT adding the brand to the title_components list
        # This is the key change to exclude brand from titles
        
        # Prioritize color band text if available (this is key for soap products)
        if attributes.get('color_band_text'):
            title_components.append({
                'type': 'primary_description',
                'text': attributes['color_band_text']
            })
            confidence += 25  # High confidence as this is typically the key product description
        
        # Add variant if available and not already in color band
        if attributes['variant'] and (not attributes.get('color_band_text') or 
                                     attributes['variant'].lower() not in attributes.get('color_band_text', '').lower()):
            title_components.append({
                'type': 'variant',
                'text': attributes['variant']
            })
            confidence += 10
        
        # Add most relevant descriptions that aren't redundant
        added_descriptions = 0
        for desc in attributes['descriptions']:
            # Skip if this text is part of already added components
            skip = False
            for comp in title_components:
                if (desc['text'].lower() in comp['text'].lower() or 
                    comp['text'].lower() in desc['text'].lower()):
                    skip = True
                    break
            
            if not skip:
                title_components.append({
                    'type': 'additional_description',
                    'text': desc['text'],
                    'relevance': desc['relevance']
                })
                confidence += desc['relevance'] * 10
                added_descriptions += 1
                
                # Limit to 1 additional description for clarity
                if added_descriptions >= 1:
                    break
        
        # Add quantity if available
        if attributes['quantity']:
            title_components.append({
                'type': 'quantity',
                'text': attributes['quantity']
            })
            confidence += 5
        
        # Construct full title - brand is excluded
        reconstructed_title = " ".join(comp['text'] for comp in title_components)
        
        # For soap products specifically, generate a standardized title without brand
        # This follows common e-commerce naming patterns for soap products
        standardized_components = []
            
        if attributes.get('color_band_text'):
            standardized_components.append(attributes['color_band_text'])
            
        elif attributes['variant']:
            if attributes['variant'].lower() != "original":
                standardized_components.append(attributes['variant'])
                standardized_components.append("Beauty Bar")
            else:
                standardized_components.append("Original Beauty Bar")
                
        # Add quantity at the end if available
        if attributes['quantity']:
            # Try to extract just the numeric part and unit
            quantity_match = re.search(r'(\d+[\.\d]*)\s*(bars?|pains?|oz|g|ml)', 
                                      attributes['quantity'], re.IGNORECASE)
            if quantity_match:
                standardized_components.append(f"{quantity_match.group(1)} {quantity_match.group(2)}")
            else:
                standardized_components.append(attributes['quantity'])
        
        standardized_title = " ".join(standardized_components)
        
        # Store the brand separately even though we're not using it in the titles
        brand = attributes['brand'] if attributes['brand'] != "Unknown" else None
        
        return {
            'components': title_components,
            'reconstructed_title': reconstructed_title,
            'standardized_title': standardized_title,
            'brand': brand,  # We still provide the brand separately
            'confidence': min(confidence, 100)  # Cap at 100%
        }
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze each product to extract and verify brand and title information.
        Uses an enhanced approach focusing on visual text containers.
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
                    
                    # Extract product attributes with enhanced approach
                    attributes = self.extract_product_attributes(
                        text_data, 
                        title,
                        identified_brand
                    )
                    
                    # Reconstruct product title from image attributes - now without brand
                    title_reconstruction = self.reconstruct_title_from_image(attributes)
                    
                    # Save results
                    result_item['attributes'] = attributes
                    result_item['reconstructed_title'] = title_reconstruction
                    
                    # Calculate title similarity
                    # Could implement more sophisticated text similarity metrics here
                    
                    # Print summary of findings
                    print(f"Product: {title[:50]}...")
                    print(f"Brand: {identified_brand} (Height: {brand_height_pct:.2f}%)")
                    print(f"Brand excluded from title reconstruction")
                    
                    if attributes['variant']:
                        print(f"Variant: {attributes['variant']}")
                    
                    if attributes['quantity']:
                        print(f"Quantity: {attributes['quantity']}")
                        
                    print(f"Reconstructed title: {title_reconstruction['reconstructed_title']}")
                    print(f"Confidence: {title_reconstruction['confidence']:.1f}%")
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


# Example usage:
"""
# Sample data for Dove soap products
sample_data = [
    {
        "brand": "Dove",
        "image": "https://example.com/dove_original.jpg",
        "price": "$13.11",
        "title": "Beauty Bar Gentle Skin Cleanser Moisturizing for Gentle Soft Skin Care Original Made With 1/4 Moisturizing Cream 3.75 oz, 14 Bars",
        "rating": "4.8 out of 5 stars"
    },
    {
        "brand": null,
        "image": "https://example.com/dove_sensitive.jpg",
        "price": "$13.11",
        "title": "Beauty Bar More Moisturizing Than Bar Soap for Softer Skin, Fragrance-Free, Hypoallergenic Beauty Bar Sensitive Skin With Gentle Cleanser 3.75 oz 14 Bars",
        "rating": "4.8 out of 5 stars"
    }
]

# Create an instance of ProductTitleExtractor
extractor = ProductTitleExtractor(sample_data)

# Process the data
processed_data = extractor.analyze()

# Print the results
import json
print(json.dumps(processed_data, indent=2))
"""