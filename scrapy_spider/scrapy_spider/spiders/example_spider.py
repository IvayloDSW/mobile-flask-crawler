import scrapy
import json
import re
from urllib.parse import urljoin

class ExampleSpider(scrapy.Spider):
    name = 'example_spider'
    allowed_domains = ['amazon.com']
    
    def __init__(self, start_url=None, *args, **kwargs):
        super(ExampleSpider, self).__init__(*args, **kwargs)
        if start_url:
            self.start_urls = [start_url]
        else:
            # Default empty list instead of hardcoded URL
            self.start_urls = []
            self.logger.warning("No start_url provided")
    
    def parse(self, response):
        """Parse Amazon product page and extract desired information."""
        product_data = {}
        
        # Get product title
        product_data['title'] = self._extract_title(response)
        
        # Get product brand
        product_data['brand'] = self._extract_brand(response)
        
        # Get product price
        product_data['price'] = self._extract_price(response)
        
        # Get product rating
        product_data['rating'] = self._extract_rating(response)
        
        # Get product image URL (focusing on mobile-friendly images)
        product_data['image_url'] = self._extract_image_url(response)
        
        # Add source URL
        product_data['url'] = response.url
        
        yield product_data
    
    def _extract_title(self, response):
        """Extract product title."""
        title = response.css('#productTitle::text').get()
        print(f"Response _extract_title: {title}")  # Debugging line
        return title.strip() if title else None
    
    def _extract_brand(self, response):
        """Extract product brand."""
        # Try different selectors for brand, starting with the one from the provided HTML snippet
        selectors = [
            'tr.po-brand td.a-span9 span::text',
            'tr.a-spacing-small.po-brand td.a-span9 span.a-size-base.po-break-word::text',
            'span.a-size-base.po-break-word::text',  # More generic version
            'a#bylineInfo::text',
            'a#brand::text',
            'a[id*="brand"]::text',
            '#bylineInfo_feature_div .a-link-normal::text',
            'tr.po-brand td.a-span9 span::text'
        ]
        
        print(f"Response _extract_brand: {selectors}")  # Debugging line
        
        for selector in selectors:
            brand = response.css(selector).get()
            if brand:
                # Clean up brand text
                brand = re.sub(r'^Visit the |^Brand: ', '', brand.strip())
                return brand
        
        return None
    
    def _extract_price(self, response):
        """Extract product price."""
        # Try different selectors for price
        selectors = [
            'span.a-price span.a-offscreen::text',
            'span#price_inside_buybox::text',
            'span.a-price-whole::text',
            '#corePriceDisplay_desktop_feature_div .a-price-whole::text',
            '#corePrice_desktop .a-price-whole::text'
        ]
        
        for selector in selectors:
            price = response.css(selector).get()
            if price:
                return price.strip()
        
        return None
    
    def _extract_rating(self, response):
        """Extract product rating."""
        # Try to get the rating text (e.g., "4.5 out of 5 stars")
        rating_text = response.css('span.a-icon-alt::text').get()
        if rating_text:
            # Extract the numerical rating using regex
            match = re.search(r'(\d+\.\d+)', rating_text)
            if match:
                return match.group(1) + " out of 5 stars"
            return rating_text.strip()
        
        # Alternative method to get rating
        rating = response.css('#acrPopover::attr(title)').get()
        if rating:
            return rating.strip()
            
        return None
    
    def _extract_image_url(self, response):
        """Extract mobile-friendly product image URL."""
        image_urls = []
        
        # Method 1: Try to extract from JSON data in scripts
        image_script = response.css('script:contains("colorImages")::text').get()
        if image_script:
            try:
                # Find the JSON data within the script
                data_match = re.search(r'var data = ({.*?});', image_script, re.DOTALL)
                if data_match:
                    json_data = json.loads(data_match.group(1))
                    
                    # Extract images from colorImages
                    if 'colorImages' in json_data and json_data['colorImages']:
                        for color, images in json_data['colorImages'].items():
                            for img in images:
                                # Prefer hiRes (high resolution) images
                                if 'hiRes' in img and img['hiRes']:
                                    image_urls.append(img['hiRes'])
                                # Fall back to large images
                                elif 'large' in img and img['large']:
                                    image_urls.append(img['large'])
            except Exception as e:
                self.logger.error(f"Error parsing image JSON: {e}")
        
        # Method 2: Try to get from the landing image
        if not image_urls:
            main_image = response.css('img#landingImage::attr(data-old-hires)').get()
            if not main_image:
                main_image = response.css('img#landingImage::attr(src)').get()
            
            if main_image:
                image_urls.append(main_image)
        
        # Method 3: Look for mobile-specific images
        if not image_urls:
            mobile_images = response.css('img.a-dynamic-image::attr(data-a-dynamic-image)').get()
            if mobile_images:
                try:
                    img_dict = json.loads(mobile_images)
                    if img_dict:
                        # Get the URL with the highest resolution
                        best_url = max(img_dict.items(), key=lambda x: x[1])[0]
                        image_urls.append(best_url)
                except Exception as e:
                    self.logger.error(f"Error parsing dynamic image data: {e}")
        
        # Return the first (best) image URL or None if none found
        return image_urls[0] if image_urls else None