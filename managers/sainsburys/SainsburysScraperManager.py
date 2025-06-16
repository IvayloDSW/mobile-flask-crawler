# from bs4 import BeautifulSoup
# from crawl4ai import *    

class SainsburysScraperManager:
    def __init__(self, url: str):
        self.url = url
        self.result = None

    # async def fetch(self):
    #     print(f"Fetching data from {self.url}")
    #     async with AsyncWebCrawler() as crawler:
    #         self.result = await crawler.arun(url=self.url)

    # def get_result(self):
    #     # print("Result fetched, parsing HTML...", self.result)  # Debugging line
    #     first_result = self.result[0] 

    #     html = first_result.html  # Get the raw HTML string
    #     print(f"HTML content fetched:...", html)  # Print the first 100 characters for debugging
    #     soup = BeautifulSoup(html, "html.parser")
        
    #     # ✅ Extract product image
    #     image_tag = soup.select_one("div.ProductImage_detailsContainer__hY1qi img")
    #     image_url = image_tag["src"] if image_tag else None

    #     # ✅ Extract price
    #     price_tag = soup.select_one('span[data-test="product-pod-price"] span')
    #     price = price_tag.get_text(strip=True) if price_tag else None

    #     # ✅ Extract rating
    #     rating_container = soup.select_one('#bv-reviews-overall-ratings-container')
    #     rating = rating_container.get_text(strip=True) if rating_container else None

    #     # ✅ Extract title (from <title> or fallback)
    #     title = soup.title.string.strip() if soup.title else "No title"

    #     # ✅ Brand is often in the title or elsewhere
    #     brand = title.split()[0] if title != "No title" else None

    #     return {
    #         "title": title,
    #         "brand": brand,
    #         "image": image_url,
    #         "price": price,
    #         "rating": rating
    #     }

   
    