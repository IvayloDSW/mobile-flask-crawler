from scrapy_splash import SplashRequest
import scrapy
import json
import re

class SainsburysSpider(scrapy.Spider):
    name = 'sainsburys_spider'
    allowed_domains = ['sainsburys.co.uk']

    def __init__(self, start_url=None, *args, **kwargs):
        super(SainsburysSpider, self).__init__(*args, **kwargs)
        self.start_url = start_url or "https://www.sainsburys.co.uk/gol-ui/product/persil-laundry-washing-liquid-detergent-non-bio-38-washes-1026l"

    def start_requests(self):
        yield SplashRequest(
            url=self.start_url,
            callback=self.parse,
            args={'wait': 2},  # wait for JS to render
        )

    def parse(self, response):
        self.logger.info("Splash-rendered response received.")

        price = response.css('span[data-testid="pd-retail-price"]::text').get()
        rating = response.css('div.ds-c-rating__stars::attr(title)').get()
        image = response.css('img[data-testid="pd-selected-image"]::attr(src)').get()

        brand = None
        json_ld = response.css('script[type="application/ld+json"]::text').get()
        if json_ld:
            try:
                data = json.loads(json_ld)
                brand = data.get('brand', {}).get('name')
            except Exception as e:
                self.logger.error(f"Error parsing brand JSON: {e}")

        yield {
            "title": response.css('title::text').get(),
            "price": price,
            "rating": rating,
            "image_url": image,
            "brand": brand,
            "url": response.url
        }
