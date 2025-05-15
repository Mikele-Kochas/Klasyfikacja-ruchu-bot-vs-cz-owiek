import subprocess
import time
import sys
import os

# --- Configuration ---
BOT_SCRIPTS = [
    "bot1_scrapy_crawler.py", # Re-enable this bot
    "bot2_selenium_browser.py",
    "bot3_api_querier.py",
    "bot4_file_downloader.py",
    "bot5_rss_reader.py",
    "bot6_form_filler.py"
]
# Duration for each bot (in seconds)
DURATION_PER_BOT = 300 # Set duration to 5 minutes (300 seconds)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Bot Sequence ---")
    
    total_start_time = time.time()
    
    for i, bot_script in enumerate(BOT_SCRIPTS):
        print(f"\n[{i+1}/{len(BOT_SCRIPTS)}] Ensuring {bot_script} runs for a total of {DURATION_PER_BOT} seconds...")
        
        if not os.path.exists(bot_script):
            print(f"Error: Script {bot_script} not found. Skipping.")
            continue
            
        bot_total_elapsed_time = 0
        session_start_time = time.time() # Track start time for this bot's overall session

        # Inner loop to ensure the bot runs for the full duration, restarting if needed
        while bot_total_elapsed_time < DURATION_PER_BOT:
            remaining_time = DURATION_PER_BOT - bot_total_elapsed_time
            print(f"  Starting/Restarting {bot_script}. Target duration for this run: {remaining_time:.2f}s")
            
            run_start_time = time.time()
            process = None
            stdout, stderr = "", "" # Initialize stdout/stderr
            try:
                process = subprocess.Popen([sys.executable, bot_script],
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE,
                                           text=True)
                
                # Wait for the process to complete or timeout
                try:
                    stdout, stderr = process.communicate(timeout=remaining_time)
                    # If communicate completes without timeout, the process finished
                    run_elapsed_time = time.time() - run_start_time
                    bot_total_elapsed_time += run_elapsed_time
                    print(f"  {bot_script} finished early (exit code {process.returncode}) after {run_elapsed_time:.2f}s.")
                    if bot_total_elapsed_time >= DURATION_PER_BOT:
                         print(f"  Total duration reached for {bot_script}.")
                         break # Exit inner loop
                    else:
                         print(f"  Restarting {bot_script} as total duration ({bot_total_elapsed_time:.2f}s) < target ({DURATION_PER_BOT}s).")
                         # Optional small delay before restart
                         time.sleep(1)
                         continue # Go to next iteration of inner loop to restart
                         
                except subprocess.TimeoutExpired:
                    # Timeout reached, bot ran for the intended remaining time (or more)
                    run_elapsed_time = time.time() - run_start_time # Actual time it ran in this instance
                    bot_total_elapsed_time += run_elapsed_time
                    print(f"  Timeout ({remaining_time:.2f}s) reached for {bot_script}. Terminating process...")
                    process.terminate()
                    try:
                        # Wait a bit for termination and capture remaining output
                        stdout, stderr = process.communicate(timeout=5) 
                    except subprocess.TimeoutExpired:
                        print(f"  Process {bot_script} did not terminate gracefully after timeout. Killing...")
                        process.kill()
                        # Try one last time to get output
                        stdout, stderr = process.communicate()
                    except Exception as comm_err_post_timeout:
                         print(f"Error communicating after timeout for {bot_script}: {comm_err_post_timeout}")
                         stdout, stderr = "","Error communicating after timeout" # Ensure stderr has content
                    
                    print(f"  {bot_script} terminated after running ~{run_elapsed_time:.2f}s in this instance.")
                    # Break the inner loop as the total time should now be >= DURATION_PER_BOT
                    break 
                except Exception as comm_err:
                    # Other communication errors (rare)
                    run_elapsed_time = time.time() - run_start_time
                    bot_total_elapsed_time += run_elapsed_time
                    print(f"  Error communicating with {bot_script}: {comm_err}. Terminating.")
                    if process.poll() is None:
                         process.terminate()
                         time.sleep(1)
                         if process.poll() is None: process.kill()
                    break # Exit inner loop on communication error

            except Exception as e:
                # Error starting the process itself
                print(f"  An error occurred starting/running {bot_script}: {e}")
                bot_total_elapsed_time = DURATION_PER_BOT # Mark as finished to prevent infinite loops
                if process and process.poll() is None:
                     print(f"  Terminating {bot_script} due to error.")
                     try: process.terminate()
                     except: pass # Ignore errors during termination
                     time.sleep(1)
                     try: 
                         if process.poll() is None: process.kill()
                     except: pass
                break # Exit inner loop on startup error
            finally:
                # Always print captured output for the completed run
                if stdout:
                    print(f"  --- Output from {bot_script} run ---\n{stdout.strip()}\n  --------------------------------")
                if stderr:
                    print(f"  --- Errors from {bot_script} run ---\n{stderr.strip()}\n  --------------------------------")
                run_elapsed_time = time.time() - run_start_time # Recalculate in case of early finish in finally
                # This check might be redundant if already handled above, but ensures loop condition updates
                if process is None or process.returncode is not None: # If process didn't start or finished
                     bot_total_elapsed_time += run_elapsed_time

        # End of inner while loop for the current bot
        session_elapsed_time = time.time() - session_start_time
        print(f"{bot_script} session finished. Total time spent: {session_elapsed_time:.2f} seconds.")
        # Small pause between bots (optional)
        time.sleep(2)

    total_elapsed_time = time.time() - total_start_time
    print(f"\n--- Bot Sequence Finished ---")
    print(f"Total execution time: {total_elapsed_time:.2f} seconds.") 