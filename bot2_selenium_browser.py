import time
import random
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, NoSuchElementException, TimeoutException
from selenium.webdriver.edge.options import Options as EdgeOptions
from urllib.parse import urlparse

# --- Configuration ---
# Start from a random choice of stable sites
INITIAL_START_URLS = [
    "https://github.com",
    "https://www.bbc.com",
    "https://www.wikipedia.org"
]
START_URL = random.choice(INITIAL_START_URLS)

# Use a diverse list for recovery
RECOVERY_URLS = [
    "https://github.com",
    "https://www.bbc.com",
    "https://www.wikipedia.org",
    "https://www.google.com",
    "https://www.onet.pl",       
    "https://httpbin.org/get", 
    "https://duckduckgo.com"  
]
RUN_DURATION_SECONDS = 90 
MIN_PAUSE_SECONDS = 0.5 # Reduced minimum pause
MAX_PAUSE_SECONDS = 4   # Slightly reduced max pause

def run_selenium_bot():
    driver = None
    start_time = time.time()
    print("Starting Bot 2: Selenium Slow Browser (Using Edge)...")
    
    try:
        # Setup Edge options
        edge_options = EdgeOptions()
        # edge_options.add_argument("--headless") # Run without opening a browser window
        edge_options.add_argument("--log-level=3") # Suppress console logs
        edge_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # Initialize Edge WebDriver using webdriver-manager
        try:
            driver = webdriver.Edge(service=EdgeService(EdgeDriverManager().install()), options=edge_options)
            driver.set_page_load_timeout(30) # Timeout for page loads
        except Exception as e:
            print(f"Error initializing Edge WebDriver: {e}")
            print("Please ensure Microsoft Edge is installed and accessible.")
            return

        driver.get(START_URL) # Start with the randomly chosen URL
        print(f"Opened start URL: {START_URL}")
        
        stuck_counter = 0
        max_stuck_count = 3
        # last_successful_url = START_URL # Not strictly needed anymore

        while time.time() - start_time < RUN_DURATION_SECONDS:
            current_url = ""
            try:
                current_url = driver.current_url
                current_title = driver.title.lower()
                print(f"Currently at: {current_url} (Title: {driver.title})")

                # --- Check for common error titles ---
                error_keywords = ["error", "not found", "problem loading", "cannot display", "unavailable"]
                if any(keyword in current_title for keyword in error_keywords):
                    print("Detected potential error page by title. Recovering...")
                    raise WebDriverException("Potential error page detected.") # Trigger recovery

                # Simulate reading time
                pause_duration = random.uniform(MIN_PAUSE_SECONDS, MAX_PAUSE_SECONDS)
                print(f"Pausing for {pause_duration:.2f} seconds...")
                time.sleep(pause_duration)

                # --- Attempt to accept cookie/privacy banners ---
                try:
                    # Common texts for acceptance buttons (add more if needed)
                    accept_texts = ['Accept', 'Accept all', 'Agree', 'I agree', 'Got it', 'OK',
                                    'Zgoda', 'Zgadzam się', 'Akceptuję', 'Rozumiem']
                    # Construct XPath selectors based on text
                    # This looks for buttons or links containing the text (case-insensitive using translate)
                    selectors = [
                        f"//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]",
                        f"//a[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}') and @href]" # Also check links
                    ]
                    
                    banner_accepted = False
                    for text in accept_texts:
                        if banner_accepted: break
                        for selector_template in selectors:
                            selector = selector_template.format(text=text)
                            try:
                                # Find elements that might be the accept button
                                accept_buttons = driver.find_elements(By.XPATH, selector)
                                for button in accept_buttons:
                                    # Check if the button is visible and clickable
                                    if button.is_displayed() and button.is_enabled():
                                        print(f"Found potential acceptance button with text '{text}'. Clicking...")
                                        driver.execute_script("arguments[0].scrollIntoView(true);", button)
                                        time.sleep(0.3)
                                        button.click()
                                        print("Acceptance button clicked.")
                                        banner_accepted = True
                                        time.sleep(1) # Wait a bit for banner to disappear/page to react
                                        break # Stop searching once clicked
                            except Exception as e_btn:
                                # Ignore errors finding specific buttons, try next selector/text
                                # print(f"Minor error searching for button '{text}': {e_btn}") 
                                pass 
                        if banner_accepted: break # Stop searching texts if already clicked
                    # if not banner_accepted:
                        # print("No common acceptance button found or clicked.")
                    
                except Exception as e_banner:
                    print(f"An error occurred during banner acceptance attempt: {e_banner}")

                # --- Find ANY potential valid links to click (not restricted to GitHub) ---
                links = driver.find_elements(By.TAG_NAME, 'a')
                valid_links = [] # Changed back from valid_github_links
                current_url_str = driver.current_url # Get current URL as string
                
                for link in links:
                    href = link.get_attribute('href')
                    # --- Stricter pre-click validation (keep this) ---
                    if not href or not isinstance(href, str):
                        continue
                    href_lower = href.lower()
                    if 'javascript:' in href_lower or 'data:' in href_lower:
                        continue 
                    
                    # --- Check if it's a valid HTTP/HTTPS link (allow any domain) ---
                    if href.startswith(('http://', 'https://')):
                        # Basic check to avoid clicking the exact same URL again
                        if href != current_url_str:
                             valid_links.append(link)
                    # Allow relative links (they stay on the same domain)
                    elif href.startswith('/') and not href.startswith('//'):
                         valid_links.append(link)
                
                if valid_links:
                    chosen_link_element = random.choice(valid_links)
                    chosen_url = chosen_link_element.get_attribute('href')
                    print(f"Attempting to click link: {chosen_url}")
                    
                    # --- Handle potential new windows ---
                    original_window = driver.current_window_handle
                    num_windows_before = len(driver.window_handles)
                    
                    # Scroll into view and click
                    driver.execute_script("arguments[0].scrollIntoView(true);", chosen_link_element)
                    time.sleep(0.5) # Small pause before click
                    chosen_link_element.click()
                    time.sleep(1.5) # Slightly longer wait to allow new windows/tabs to potentially open

                    # Check if a new window was opened
                    num_windows_after = len(driver.window_handles)
                    if num_windows_after > num_windows_before:
                        print(f"Detected {num_windows_after - num_windows_before} new window(s) opening.")
                        all_windows = driver.window_handles
                        for window_handle in all_windows:
                            if window_handle != original_window:
                                try:
                                    print(f"Switching to and closing new window: {window_handle}")
                                    driver.switch_to.window(window_handle)
                                    driver.close()
                                except Exception as e_win:
                                    print(f"Error closing new window: {e_win}")
                        # Switch back to the original window
                        driver.switch_to.window(original_window)
                        print("Switched back to original window. Treating as navigation error.")
                        raise WebDriverException("New window opened by script") # Trigger recovery

                    # --- Check if landed on data: URL (in the current window) ---
                    current_url_after_click = driver.current_url
                    if current_url_after_click.startswith('data:'):
                        print(f"Landed on a 'data:' URL ({current_url_after_click[:30]}...). Recovering...")
                        raise WebDriverException("Landed on data: URL") # Trigger recovery
                        
                    # If no new window and not data: URL, reset stuck counter
                    stuck_counter = 0 
                else:
                    # --- Stuck handling (uses RECOVERY_URLS again) ---
                    print("No suitable outgoing links found on page.")
                    stuck_counter += 1
                    if stuck_counter >= max_stuck_count:
                        recovery_url = random.choice(RECOVERY_URLS)
                        print(f"Stuck for {stuck_counter} attempts. Forcing navigation to recovery URL: {recovery_url}")
                        try:
                            driver.get(recovery_url)
                            stuck_counter = 0 # Reset after forcing
                        except Exception as e_nav:
                            print(f"Failed to navigate to recovery URL {recovery_url} after error: {e_nav}. Exiting bot.")
                            break # Exit loop if even recovery navigation fails
                    else:
                        # Try going back first, then to a random recovery URL as fallback
                        print("Attempting to navigate back...")
                        try: 
                            driver.back()
                        except Exception:
                            print(f"Failed to navigate back. Attempting recovery URL: {recovery_url}")
                            try: 
                                driver.get(recovery_url)
                            except Exception as e_nav_start:
                                print(f"Failed to navigate to recovery URL {recovery_url}: {e_nav_start}. Exiting bot.")
                                break # Exit loop if navigation fails
                    time.sleep(2) # Longer sleep after error
            
            # --- Consolidated Error Handling (uses RECOVERY_URLS) ---
            except Exception as e:
                print(f"An error occurred: {type(e).__name__}. Attempting recovery...")
                stuck_counter += 1
                recovery_url = random.choice(RECOVERY_URLS) # Choose recovery URL
                if stuck_counter >= max_stuck_count:
                     print(f"Multiple errors or stuck for {stuck_counter} attempts. Forcing navigation to recovery URL: {recovery_url}")
                     try:
                         driver.get(recovery_url)
                         stuck_counter = 0 # Reset after forcing
                     except Exception as e_nav:
                         print(f"Failed to navigate to recovery URL {recovery_url} after error: {e_nav}. Exiting bot.")
                         break # Exit loop if even recovery navigation fails
                else:
                    # Try going back first, then to a random recovery URL as fallback
                    print("Attempting to navigate back...")
                    try: 
                        driver.back()
                    except Exception:
                        print(f"Failed to navigate back. Attempting recovery URL: {recovery_url}")
                        try: 
                            driver.get(recovery_url)
                        except Exception as e_nav_start:
                            print(f"Failed to navigate to recovery URL {recovery_url}: {e_nav_start}. Exiting bot.")
                            break # Exit loop if navigation fails
                    time.sleep(2) # Longer sleep after error

    except Exception as e:
        print(f"An error occurred during Selenium bot execution: {e}")
    finally:
        if driver:
            print("Closing Selenium WebDriver...")
            driver.quit()
        print("Bot 2 finished.")

if __name__ == "__main__":
    run_selenium_bot() 