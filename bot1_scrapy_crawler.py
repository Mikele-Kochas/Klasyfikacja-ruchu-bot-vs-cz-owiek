import scrapy
import random
import argparse
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings

class RandomCrawlerSpider(scrapy.Spider):
    name = "random_crawler"
    visited_urls = set()

    def __init__(self, url=None, *args, **kwargs):
        super(RandomCrawlerSpider, self).__init__(*args, **kwargs)
        if url is None:
            self.start_url = "https://www.onet.pl" # Default example
        else:
            self.start_url = url
        self.log(f"Starting crawl from: {self.start_url}")

    def start_requests(self):
        yield scrapy.Request(self.start_url, self.parse, dont_filter=True) # Start with dont_filter just in case

    def parse(self, response):
        self.visited_urls.add(response.url)
        self.log(f"Visited: {response.url}")

        links = response.css('a::attr(href)').getall()
        valid_links = []
        for link in links:
            absolute_link = response.urljoin(link)
            if absolute_link.startswith(('http://', 'https://')) and '#' not in absolute_link and not absolute_link.startswith('javascript:'):
                valid_links.append(absolute_link)

        if valid_links:
            next_url = random.choice(valid_links)
            self.log(f"Following random link: {next_url}")
            yield scrapy.Request(next_url, callback=self.parse, dont_filter=True)
        else:
            self.log(f"No valid outgoing links found on: {response.url}. Returning to start URL.")
            yield scrapy.Request(self.start_url, callback=self.parse, dont_filter=True)

        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", default="https://www.onet.pl",
                        help="The starting URL for the crawler")
    args = parser.parse_args()

    settings = Settings()
    settings['ROBOTSTXT_OBEY'] = False
    settings['LOG_LEVEL'] = 'INFO'
    settings['COOKIES_ENABLED'] = True
    settings['HTTPERROR_ALLOWED_CODES'] = [404, 999]

    process = CrawlerProcess(settings)
    process.crawl(RandomCrawlerSpider, url=args.url)
    process.start() # This will now run until killed externally