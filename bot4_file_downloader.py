import requests
import time
import random
import os

# --- Configuration ---
# URLs of files to download
FILE_URLS = [
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", # ~1MB PDF
    "https://speed.hetzner.de/100MB.bin", # 100MB binary file
    "https://picsum.photos/1200/800", # Random large image
    # Add more URLs if needed
]
DOWNLOAD_DIR = "downloaded_files"
# Remove internal duration limit
# RUN_DURATION_SECONDS = 90 
MIN_DELAY_SECONDS = 5
MAX_DELAY_SECONDS = 15

def run_file_downloader():
    start_time = time.time()
    print("Starting Bot 4: File Downloader...")
    session = requests.Session() # Use a session
    session.headers.update({'User-Agent': 'SimpleFileDownloaderBot/1.0'})

    # Create download directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created directory: {DOWNLOAD_DIR}")

    try:
        download_count = 0
        # Change to an infinite loop
        while True:
            # Choose a random file URL to download
            file_url = random.choice(FILE_URLS)
            
            # Create a filename (simple approach)
            filename = os.path.join(DOWNLOAD_DIR, f"download_{download_count}_{os.path.basename(file_url) or 'random_image.jpg'}")
            download_count += 1
            
            try:
                print(f"Attempting to download: {file_url} to {filename}")
                # Use stream=True to handle potentially large files efficiently
                with session.get(file_url, stream=True, timeout=60) as response: # 60 sec timeout for download
                    response.raise_for_status() # Check for HTTP errors
                    
                    # Write the file content chunk by chunk
                    with open(filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192): 
                            f.write(chunk)
                    print(f"Successfully downloaded {file_url} to {filename} (Size: {os.path.getsize(filename)} bytes)")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_url}: {e}. Skipping to next file.")
                # Continue to the next iteration below
            except Exception as e:
                 print(f"An unexpected error occurred during download of {file_url}: {e}. Skipping to next file.")
                 # Continue to the next iteration below

            # Wait for a random delay before next download (Always sleep)
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            # print(f"Waiting for {delay:.2f} seconds...")
            time.sleep(delay)
            
    except Exception as e:
         print(f"An error occurred during File Downloader execution: {e}")
    finally:
        print("Bot 4 finished.")

if __name__ == "__main__":
    run_file_downloader() 