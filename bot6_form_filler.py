import requests
import time
import random
import string

# --- Configuration ---
FORM_URL = "https://httpbin.org/post" # Target URL that accepts POST requests
MIN_DELAY_SECONDS = 4
MAX_DELAY_SECONDS = 10
# Remove internal duration limit
# RUN_DURATION_SECONDS = 90 

# Function to generate random data
def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_random_email():
    return f"{generate_random_string(8)}@{generate_random_string(5)}.com"

def run_form_filler():
    start_time = time.time()
    print("Starting Bot 6: Form Filler...")
    session = requests.Session() # Use a session
    session.headers.update({'User-Agent': 'SimpleFormFillerBot/1.0'})

    try:
        # Change to an infinite loop
        while True:
            # Prepare random form data
            form_data = {
                'customer': generate_random_string(12),
                'email': generate_random_email(),
                'comments': generate_random_string(50),
                'random_number': random.randint(1, 1000)
            }
            
            try:
                print(f"Submitting form data to: {FORM_URL}")
                # print(f"Data: {form_data}")
                response = session.post(FORM_URL, data=form_data, timeout=15)
                response.raise_for_status() # Check for HTTP errors
                
                # httpbin.org/post returns the posted data in JSON format
                # print(f"Response Status: {response.status_code}")
                # print(f"Response Body (form part): {response.json().get('form')}")
                print(f"Successfully submitted form to {FORM_URL} (Status: {response.status_code})")

            except requests.exceptions.RequestException as e:
                print(f"Error submitting form to {FORM_URL}: {e}. Skipping to next submission.")
                # Continue below
            except Exception as e:
                 print(f"An unexpected error occurred during form submission to {FORM_URL}: {e}. Skipping to next submission.")
                 # Continue below

            # Wait for a random delay before next submission (Always sleep)
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            # print(f"Waiting for {delay:.2f} seconds...")
            time.sleep(delay)
            
    except Exception as e:
         print(f"An error occurred during Form Filler execution: {e}")
    finally:
        print("Bot 6 finished.")

if __name__ == "__main__":
    run_form_filler() 