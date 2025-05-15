import requests
import feedparser
import time
import random

# --- Configuration ---
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.wired.com/feed/rss",
    "http://rss.cnn.com/rss/cnn_topstories.rss"
]
MIN_DELAY_SECONDS = 15
MAX_DELAY_SECONDS = 30
# Remove internal duration limit
# RUN_DURATION_SECONDS = 90 

def run_rss_reader():
    start_time = time.time()
    print("Starting Bot 5: RSS Reader...")
    session = requests.Session() # Use a session
    session.headers.update({'User-Agent': 'SimpleRSSReaderBot/1.0'})

    try:
        # Change to an infinite loop
        while True:
            # Choose a random feed to check
            feed_url = random.choice(RSS_FEEDS)
            
            try:
                print(f"Fetching RSS feed: {feed_url}")
                # Fetch the feed content using requests
                response = session.get(feed_url, timeout=15)
                response.raise_for_status()
                
                # Parse the feed content
                feed_data = feedparser.parse(response.content)
                
                # Check for parsing errors (bozo may indicate non-well-formed feed)
                if feed_data.bozo:
                    print(f"Warning: Feed {feed_url} may not be well-formed. Bozo exception: {feed_data.bozo_exception}")
                
                # Print some basic info about the feed
                feed_title = feed_data.feed.get('title', 'N/A')
                num_entries = len(feed_data.entries)
                print(f"Successfully parsed feed '{feed_title}' from {feed_url}. Found {num_entries} entries.")

            except requests.exceptions.RequestException as e:
                print(f"Error fetching feed {feed_url}: {e}. Skipping to next feed.")
                # Continue below
            except Exception as e:
                 print(f"An unexpected error occurred during RSS processing for {feed_url}: {e}. Skipping to next feed.")
                 # Continue below

            # Wait for a random delay before checking next feed (Always sleep)
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            # print(f"Waiting for {delay:.2f} seconds...")
            time.sleep(delay)
            
    except Exception as e:
         print(f"An error occurred during RSS Reader execution: {e}")
    finally:
        print("Bot 5 finished.")

if __name__ == "__main__":
    run_rss_reader() 