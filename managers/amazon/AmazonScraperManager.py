import os
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional

class AmazonScraperManager:
    """
    A class to manage Amazon product scraping using Scrapy.
    """
    
    def __init__(self, scrapy_project_path: Optional[str] = None, output_file_path: Optional[str] = None):
        """
        Initialize the scraper manager.
        
        Args:
            scrapy_project_path: Path to the Scrapy project directory
            output_file_path: Path to save the output JSON
        """
        self.scrapy_project_path = scrapy_project_path or os.path.join(os.getcwd(), 'scrapy_spider')
        self.output_file_path = output_file_path or os.path.join(os.getcwd(), 'output.json')
        self.logger = logging.getLogger(__name__)
        
    def scrape_product(self, url: str) -> Dict[str, Any]:
        """
        Scrape product information from the given URL.
        
        Args:
            url: The Amazon product URL to scrape
            
        Returns:
            A dictionary containing the scraped product data or error information
        """
        # Ensure the URL is provided
        if not url:
            return {'error': 'Missing URL parameter'}
            
        self.logger.info(f"Scraping URL: {url}")
        
        # Remove old output file if it exists
        if os.path.exists(self.output_file_path):
            try:
                os.remove(self.output_file_path)
            except Exception as e:
                self.logger.warning(f"Failed to remove old output file: {e}")
        
        # Construct the Scrapy command
        command = [
            'scrapy', 'crawl', 'example_spider',
            '-a', f'start_url={url}',
            '-o', self.output_file_path
        ]
        
        try:
            # Run the Scrapy command
            result = subprocess.run(
                command,
                cwd=self.scrapy_project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Check if the output file exists
            if os.path.exists(self.output_file_path):
                with open(self.output_file_path, 'r') as file:
                    raw_data = file.read()
                    
                # Parse the JSON data
                try:
                    data = json.loads(raw_data)
                    # If it's a list with one item (typical Scrapy output), return the first item
                    if isinstance(data, list) and len(data) > 0:
                        return {'success': True, 'data': data[0]}
                    return {'success': True, 'data': data}
                except json.JSONDecodeError as e:
                    return {'success': False, 'error': f'Failed to parse JSON: {str(e)}', 'raw_data': raw_data}
            else:
                return {'success': False, 'error': 'Output file not found'}
                
        except subprocess.CalledProcessError as e:
            # Handle errors in the subprocess
            return {'success': False, 'error': f'Scrapy command failed: {e.stderr}'}
        except Exception as e:
            # Handle other exceptions
            return {'success': False, 'error': f'An error occurred: {str(e)}'}
            
    def batch_scrape(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple product URLs.
        
        Args:
            urls: List of Amazon product URLs to scrape
            
        Returns:
            A list of dictionaries containing the scraped product data
        """
        results = []
        for url in urls:
            result = self.scrape_product(url)
            result['url'] = url
            results.append(result)
        return results


