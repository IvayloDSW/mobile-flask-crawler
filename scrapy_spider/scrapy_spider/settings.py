# Scrapy settings for scrapy_spider project
BOT_NAME = 'scrapy_spider'

SPIDER_MODULES = ['scrapy_spider.spiders']
NEWSPIDER_MODULE = 'scrapy_spider.spiders'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'


LOG_LEVEL = 'ERROR'  # Or 'CRITICAL' for even less output


ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 32

# Configure a delay for requests to the same website (default: 0)
DOWNLOAD_DELAY = 3
