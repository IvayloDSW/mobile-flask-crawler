
from google.cloud import vision


class AnalyzeBrandTitle:
    def __init__(self, data):
        self.data = data
        
    def extract_text_from_url(self, image_url):
        # Initialize the Google Cloud Vision client
        client = vision.ImageAnnotatorClient()

        # Load the image from the URL
        image = vision.Image()
        image.source.image_uri = image_url

        # Perform text detection on the image
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f"{response.error.message}")

        # Extract the detected text
        detected_text = texts[0].description if texts else ''
        return detected_text    

    def analyze(self):
        for item in self.data:
            if isinstance(item, dict):
                # Assuming the dictionary has a 'title' key
                title = item.get('title', '')
                brand = item.get('brand', '')
                image = item.get('image', '')
                if image:
                    # Extract text from the image URL
                    detected_text = self.extract_text_from_url(image)
                    print(f"Detected text from image: {detected_text}")
                    item['detected_text'] = detected_text
                if brand:
                    # Perform analysis on the brand
                    # print(f"Analyzing brand: {brand}")
                    # Add your analysis logic here
                    # For example, you could modify the brand or extract information
                    item['analyzed_brand'] = f"Analyzed: {brand}"
                else:
                    print("No brand found in item.")
                    
                if title:
                    # Perform analysis on the title
                    # print(f"Analyzing title: {title}")
                    # Add your analysis logic here
                    # For example, you could modify the title or extract information
                    item['analyzed_title'] = f"Analyzed: {title}"
                else:
                    print("No title found in item.")
            else:
                print("Item is not a dictionary.")
                
            return item    
    
    def get_title(self):
        return self.data