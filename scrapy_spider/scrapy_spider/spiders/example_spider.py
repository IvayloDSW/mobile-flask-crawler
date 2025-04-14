# import scrapy

# class ExampleSpider(scrapy.Spider):
#     name = 'example_spider'
#     allowed_domains = ['amazon.com']
#     start_urls = [
#         'https://www.amazon.com/TOZO-Cancelling-Waterproof-Bluetooth-Headphones/dp/B0DG8NMPSH/'
#     ]

#     def parse(self, response):
#         title = response.css('span#productTitle::text').get().strip()
#         image_url = response.css('img#landingImage::attr(src)').get()
#         yield {
#             'title': title,
#             'image_url': image_url
#         }

import scrapy

class ExampleSpider(scrapy.Spider):
    name = 'example_spider'
    allowed_domains = ['amazon.com']

    def __init__(self, start_url=None, *args, **kwargs):
        super(ExampleSpider, self).__init__(*args, **kwargs)
        if start_url:
            self.start_urls = [start_url]
        else:
            self.start_urls = [
                'https://www.amazon.com/TOZO-Cancelling-Waterproof-Bluetooth-Headphones/dp/B0DG8NMPSH/'
            ]

    def parse(self, response):
        title = response.css('span#productTitle::text').get().strip()
        image_url = response.css('img#landingImage::attr(src)').get()
        yield {
            'title': title,
            'image_url': image_url
        }

