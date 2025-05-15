import requests
import time
import random

# --- Configuration ---
API_URLS = [
    "https://jsonplaceholder.typicode.com/posts/", # Will append random post ID (1-100)
    "https://api.quotable.io/random",
    "https://httpbin.org/get" # Simple GET endpoint
]
MIN_DELAY_SECONDS = 3
MAX_DELAY_SECONDS = 8
# Remove internal duration limit
# RUN_DURATION_SECONDS = 90 

def run_api_querier():
    start_time = time.time()
    print("Starting Bot 3: API Querier...")
    session = requests.Session() # Use a session for potential efficiency
    session.headers.update({'User-Agent': 'SimpleAPIQuerierBot/1.0'})

    try:
        # Change to an infinite loop
        while True:
            # Choose a random API endpoint
            api_choice = random.choice(API_URLS)
            target_url = api_choice

            try:
                # Append random ID if it's the posts URL
                if "jsonplaceholder.typicode.com/posts/" in api_choice:
                    post_id = random.randint(1, 100)
                    target_url = f"{api_choice}{post_id}"
                
                print(f"Querying API: {target_url}")
                response = session.get(target_url, timeout=10) # 10 second timeout
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                
                # Optional: print status or part of the content
                # print(f"Status Code: {response.status_code}")
                # print(f"Response (first 100 chars): {response.text[:100]}...")
                print(f"Successfully queried {target_url} (Status: {response.status_code})")

            except requests.exceptions.RequestException as e:
                print(f"Error querying {target_url}: {e}. Skipping to next API.")
                # Continue to the next iteration to pick a different API
                # Ensure delay still happens even on error
                # continue # Mistake: continue would skip the delay. Delay should happen.
            except Exception as e:
                 print(f"An unexpected error occurred during request for {target_url}: {e}. Skipping to next API.")
                 # Continue to the next iteration
                 # continue # Mistake: continue would skip the delay. Delay should happen.

            # Wait for a random delay (Always sleep, even after error)
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            # print(f"Waiting for {delay:.2f} seconds...")
            time.sleep(delay)
            
    except Exception as e:
         print(f"An error occurred during API Querier execution: {e}")
    finally:
        print("Bot 3 finished.")

if __name__ == "__main__":
    run_api_querier() 