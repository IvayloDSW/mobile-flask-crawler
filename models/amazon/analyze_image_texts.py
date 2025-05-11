from google.cloud import vision
import requests
from io import BytesIO
from PIL import Image, ImageStat
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import math
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from models.amazon.title_predicted import TitlePredictor
from models.amazon.sefl_title_predictor import SelfLearningTitlePredictor

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color code."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def get_dominant_color(pixels: np.ndarray, n_clusters: int = 1) -> Tuple[int, int, int]:
    """
    Find the dominant color in a set of pixels using K-means clustering.
    Returns the RGB values of the most common color.
    """
    if len(pixels) == 0:
        return (0, 0, 0)
        
    # Reshape pixels to be a list of RGB points
    pixels = pixels.reshape(-1, 3)
    
    # Use smaller sample for efficiency if there are many pixels
    pixel_count = len(pixels)
    sample_size = min(pixel_count, 1000)
    
    if pixel_count > sample_size:
        indices = np.random.choice(pixel_count, sample_size, replace=False)
        pixels = pixels[indices]
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(pixels)
    
    # Get the most common color (largest cluster)
    counts = np.bincount(kmeans.labels_)
    dominant_cluster = np.argmax(counts)
    center = kmeans.cluster_centers_[dominant_cluster]
    
    # Return as integer RGB tuple
    return tuple(map(int, center))

def adaptive_threshold(crop: Image.Image) -> int:
    """
    Determine an adaptive threshold based on the image histogram.
    """
    gray = np.array(crop.convert('L'))
    
    # Try Otsu's method for adaptive thresholding
    try:
        thresh = threshold_otsu(gray)
        return int(thresh)
    except:
        # Fallback to a reasonable default if Otsu's method fails
        return 128

def split_text_bg_colors(crop: Image.Image) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """
    Separate a cropped row into background vs. text using adaptive thresholding
    and advanced color analysis.
    Returns (background_rgb, text_rgb).
    """
    # Apply mild Gaussian blur to reduce noise
    crop_array = np.array(crop.convert('RGB'))
    blurred_rgb = gaussian_filter(crop_array, sigma=[0.5, 0.5, 0], mode='nearest')
    blurred_rgb = blurred_rgb.astype(np.uint8)
    
    # Convert to grayscale for text/background separation
    gray = np.array(Image.fromarray(blurred_rgb).convert('L'))
    
    # Get adaptive threshold
    threshold = adaptive_threshold(crop)
    
    # Create masks based on brightness
    text_mask = gray < threshold
    bg_mask = gray >= threshold
    
    # Expand text mask slightly to include anti-aliased pixels
    text_mask = binary_dilation(text_mask, iterations=1)
    # Update bg_mask to be the inverse of text_mask
    bg_mask = ~text_mask
    
    # Fallback if either mask is empty or too small
    if text_mask.sum() < 10 or bg_mask.sum() < 10:
        avg = tuple(map(int, np.mean(crop_array, axis=(0,1))))
        return avg, avg
    
    # Get pixels for each category
    text_pixels = blurred_rgb[text_mask]
    bg_pixels = blurred_rgb[bg_mask]
    
    # Use K-means clustering to find dominant colors
    bg_color = get_dominant_color(bg_pixels, n_clusters=1)
    text_color = get_dominant_color(text_pixels, n_clusters=1)
    
    return bg_color, text_color

def linearize_channel(c: float) -> float:
    """Linearize a color channel according to sRGB standard."""
    c = c / 255.0
    return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055) ** 2.4

def relative_luminance(rgb: Tuple[int,int,int]) -> float:
    """Calculate relative luminance according to WCAG 2.0."""
    r, g, b = rgb
    R = linearize_channel(r)
    G = linearize_channel(g)
    B = linearize_channel(b)
    return 0.2126*R + 0.7152*G + 0.0722*B

def contrast_ratio(c1: Tuple[int,int,int], c2: Tuple[int,int,int]) -> float:
    """Calculate contrast ratio between two colors according to WCAG 2.0."""
    L1 = relative_luminance(c1)
    L2 = relative_luminance(c2)
    light, dark = max(L1, L2), min(L1, L2)
    return (light + 0.05) / (dark + 0.05)



class AnalyzeImageTexts:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.cache = {}  # Simple cache for downloaded images

    def _download_image(self, url: str) -> Image.Image:
        """Download image from URL with caching."""
        if url in self.cache:
            return self.cache[url]
            
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        self.cache[url] = img
        return img

    def extract_text_from_url(self, image_url: str) -> Dict[str, Any]:
        """Extract text from image using Google Vision API."""
        client = vision.ImageAnnotatorClient()
        image = vision.Image()
        image.source.image_uri = image_url
        resp = client.text_detection(image=image)
        if resp.error.message:
            raise Exception(f"Google Vision API error: {resp.error.message}")

        full_text = ""
        w = h = 0
        if resp.text_annotations:
            ann0 = resp.text_annotations[0]
            full_text = ann0.description.strip()
            verts = ann0.bounding_poly.vertices
            w = max(v.x for v in verts)
            h = max(v.y for v in verts)

        blocks = []
        for ann in resp.text_annotations[1:]:
            verts = ann.bounding_poly.vertices
            tl, br = verts[0], verts[2]
            bw, bh = br.x - tl.x, br.y - tl.y
            blocks.append({
                'text': ann.description.strip(),
                'x': tl.x, 'y': tl.y,
                'width': bw, 'height': bh,
                'height_percentage': (bh / h * 100) if h else 0,
                'y_position_percentage': (tl.y / h * 100) if h else 0
            })
        blocks.sort(key=lambda b: b['y'])
        return {
            'full_text': full_text,
            'text_blocks': blocks,
            'image_dimensions': {'width': w, 'height': h}
        }

    def identify_text_rows(self, text_blocks: List[Dict[str, Any]], image_height: int) -> List[Dict[str, Any]]:
        """Group text blocks into rows based on vertical positioning."""
        if not text_blocks:
            return []
            
        # Use adaptive threshold based on image height
        threshold_percentage = 2  # 2% of image height
        thresh = (threshold_percentage/100) * image_height
        
        rows = [{
            'blocks': [text_blocks[0]],
            'top': text_blocks[0]['y'],
            'bottom': text_blocks[0]['y'] + text_blocks[0]['height']
        }]
        
        for blk in text_blocks[1:]:
            top, bot = blk['y'], blk['y'] + blk['height']
            cur = rows[-1]
            
            # Check if block is close enough to current row
            if abs(top - cur['top']) < thresh or abs(bot - cur['bottom']) < thresh or \
               (top > cur['top'] and bot < cur['bottom']) or (cur['top'] > top and cur['bottom'] < bot):
                cur['blocks'].append(blk)
                cur['top'] = min(cur['top'], top)
                cur['bottom'] = max(cur['bottom'], bot)
            else:
                rows.append({'blocks': [blk], 'top': top, 'bottom': bot})
        
        # Calculate row metrics
        for r in rows:
            r['height'] = r['bottom'] - r['top']
            r['blocks'].sort(key=lambda b: b['x'])
            r['combined_text'] = ' '.join(b['text'] for b in r['blocks'])
            r['height_percentage'] = (r['height'] / image_height * 100) if image_height else 0
            r['position_percentage'] = (r['top'] / image_height * 100) if image_height else 0
            
        return rows

    def calculate_font_size_category(self, height_percentage: float) -> str:
        """Categorize text size based on height percentage."""
        if height_percentage > 10:
            return "very large"
        elif height_percentage > 5:
            return "large"
        elif height_percentage > 2:
            return "medium"
        elif height_percentage > 1:
            return "small"
        else:
            return "very small"

    def predict_product_title(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the product title from extracted text rows.
        
        Args:
            item_data: Dict containing extracted_text_rows, brand, and original title
            
        Returns:
            Dict with predicted_title and related metrics
        """
       
        predictor = TitlePredictor(item_data)
        title_prediction = predictor.predict_title()
        
        # model_predictor = SelfLearningTitlePredictor(item_data)
        # model_title_prediction = model_predictor.predict()
        # print(f"Title prediction: {title_prediction}")
        
        # Add the predicted title to the results
        results = {
            **item_data,
            'predicted_title': title_prediction['predicted_title'],
            'title_prediction': title_prediction,
            'model_title_prediction': {},
            # 'model_title_prediction': model_title_prediction,
        }
        
        return results

    def analyze(self) -> List[Dict[str, Any]]:
        """Main analysis method to process all images."""
        results = []
        index = 0
        
        for item in self.data:
            url = item.get('image')
            if not url:
                results.append({**item, 'error': 'No image URL'})
                continue
                
            try:
                # Extract text using Google Vision API
                index += 1
                td = self.extract_text_from_url(url)
                rows = self.identify_text_rows(td['text_blocks'], td['image_dimensions']['height'])
                
                # Download image for color analysis
                img = self._download_image(url)
                
                extracted_rows = []
                for r in rows:
                    # Add padding to crop to ensure we get the full text area
                    padding = max(2, int(r['height'] * 0.05))  # 5% padding or at least 2px
                    
                    left = max(0, min(b['x'] for b in r['blocks']) - padding)
                    right = min(td['image_dimensions']['width'], 
                               max(b['x'] + b['width'] for b in r['blocks']) + padding)
                    top = max(0, r['top'] - padding)
                    bot = min(td['image_dimensions']['height'], r['bottom'] + padding)
                    
                    # Ensure valid crop dimensions
                    if right <= left or bot <= top:
                        continue
                        
                    # Crop the text row from the image
                    crop = img.crop((left, top, right, bot))
                    
                    # Extract colors and calculate contrast
                    bg_rgb, text_rgb = split_text_bg_colors(crop)
                    cr = contrast_ratio(bg_rgb, text_rgb)
                    visible = cr >= 4.5  # WCAG AA standard for normal text
                    
                    size_category = self.calculate_font_size_category(r['height_percentage'])
                    
                    extracted_rows.append({
                        'text': r['combined_text'],
                        'height': r['height'],
                        'height_percentage': round(r['height_percentage'], 2),
                        'position_percentage': round(r['position_percentage'], 2),
                        'size_category': size_category,
                        'background_color_rgb': bg_rgb,
                        'background_color_hex': rgb_to_hex(bg_rgb),
                        'text_color_rgb': text_rgb,
                        'text_color_hex': rgb_to_hex(text_rgb),
                        'contrast_ratio': round(cr, 2),
                        'visible': visible,
                        'wcag_aa_pass': cr >= 4.5,  # WCAG AA standard
                        'wcag_aaa_pass': cr >= 7.0  # WCAG AAA standard
                    })

                # Find the largest and most prominent text rows
                if extracted_rows:
                    largest = max(extracted_rows, key=lambda x: x['height_percentage'], default=None)
                    
                    # Try to find the title/heading text (usually larger and near the top)
                    top_third_rows = [r for r in extracted_rows 
                                     if r['position_percentage'] < 33 and r['height_percentage'] > 1.5]
                    
                    prominent_text = largest if not top_third_rows else max(
                        top_third_rows, key=lambda x: x['height_percentage'])
                    
                    largest_info = {
                        'text': largest['text'],
                        'height_percentage': largest['height_percentage'],
                        'contrast_ratio': largest['contrast_ratio'],
                        'visible': largest['visible'],
                        'size_category': largest['size_category']
                    } if largest else {}
                    
                    prominent_info = {
                        'text': prominent_text['text'],
                        'height_percentage': prominent_text['height_percentage'],
                        'position_percentage': prominent_text['position_percentage'],
                        'contrast_ratio': prominent_text['contrast_ratio'],
                        'visible': prominent_text['visible'],
                        'size_category': prominent_text['size_category']
                    } if prominent_text else {}
                else:
                    largest_info = {}
                    prominent_info = {}

                item_result = {
                    **item,
                    "id": f"item_{index}",
                    'image_dimensions': td['image_dimensions'],
                    'extracted_text_rows': extracted_rows,
                    'largest_text_row': largest_info,
                    'prominent_text': prominent_info,
                    'accessibility': {
                        'has_text': len(extracted_rows) > 0,
                        'all_text_visible': all(r['visible'] for r in extracted_rows) if extracted_rows else False,
                        'visible_text_percentage': round(
                            sum(1 for r in extracted_rows if r['visible']) / len(extracted_rows) * 100 
                            if extracted_rows else 0, 1
                        )
                    }
                }
                
                # Add title prediction
                if 'title' in item:
                    item_result = self.predict_product_title(item_result)
                
                results.append(item_result)
            except Exception as e:
                print(f"Error processing item {index}: {e}")
                results.append({**item, 'error': str(e)})
                
        return results