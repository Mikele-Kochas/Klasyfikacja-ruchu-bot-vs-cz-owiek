import pandas as pd
import numpy as np
from scapy.all import rdpcap, PcapReader
from datetime import datetime, timedelta
import warnings
import math
import os

# Ignore specific Scapy warnings if they appear
warnings.filterwarnings("ignore", category=UserWarning, module='scapy.runtime')

# --- Configuration ---
PCAP_FILES = {
    r"C:\Users\mikel\Desktop\Projekt Cyberbezpieczeństwo\human_traffic.pcapng": 0, # Label 0 for human
    r"C:\Users\mikel\Desktop\Projekt Cyberbezpieczeństwo\bot_traffic.pcapng": 1    # Label 1 for bot
}
OUTPUT_CSV = "labled_trafic.csv"
TIME_WINDOW_SECONDS = 5

# --- Feature Calculation ---

def calculate_stats(data):
    """Calculates statistics for a list of numbers (packet sizes or inter-arrival times)."""
    if not data:
        return 0, 0, 0, 0 # mean, median, stddev, variance
    if len(data) == 1:
        val = data[0]
        return val, val, 0, 0
    
    mean = np.mean(data)
    median = np.median(data)
    stddev = np.std(data)
    variance = np.var(data)
    return mean, median, stddev, variance

def process_pcap(filename, label, window_size_seconds):
    """Processes a single pcap file and extracts features for time windows."""
    features_list = []
    packets_in_window = []
    window_start_time = None
    local_ip = None # We'll try to determine this from the first packet

    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found. Skipping.")
        return []

    print(f"Processing {filename}...")
    try:
        # Use PcapReader for potentially large files
        with PcapReader(filename) as pcap_reader:
            for i, packet in enumerate(pcap_reader):
                # Determine local IP from the first IP packet's source
                if local_ip is None and 'IP' in packet:
                    local_ip = packet['IP'].src
                    print(f"Determined local IP as: {local_ip}")

                if 'IP' not in packet or 'TCP' not in packet: # Consider only TCP/IP packets
                    continue
                
                # Scapy uses Decimal timestamps, convert to float
                timestamp = float(packet.time)
                packet_len = len(packet)
                
                # Determine direction
                direction = None
                if local_ip:
                    if packet['IP'].src == local_ip:
                        direction = 'sent'
                    elif packet['IP'].dst == local_ip:
                        direction = 'received'
                    # else: packet is not to/from local IP (e.g., broadcast, other traffic), ignore for directional stats? For now, let's include it in totals but not directional.
                
                if window_start_time is None:
                    window_start_time = timestamp

                # Check if packet belongs to the current window
                if timestamp < window_start_time + window_size_seconds:
                    packets_in_window.append({'timestamp': timestamp, 'len': packet_len, 'direction': direction})
                else:
                    # --- Process the completed window ---
                    if packets_in_window:
                        features = extract_features(packets_in_window, window_start_time, window_size_seconds, label)
                        features_list.append(features)
                    
                    # Start the new window
                    window_start_time = window_start_time + window_size_seconds
                    # Handle gaps: advance window start time until it includes the current packet
                    while timestamp >= window_start_time + window_size_seconds:
                         # Add empty window features if needed, or just advance time
                         # For simplicity, just advance time. Alternatively add rows with zeros.
                         # Let's add rows with zeros for empty windows.
                         empty_features = create_empty_features(window_start_time, window_size_seconds, label)
                         features_list.append(empty_features)
                         window_start_time += window_size_seconds

                    packets_in_window = [{'timestamp': timestamp, 'len': packet_len, 'direction': direction}]

            # Process the last window
            if packets_in_window:
                features = extract_features(packets_in_window, window_start_time, window_size_seconds, label)
                features_list.append(features)
            
            # Add empty windows at the end if the capture ended before a full window duration? Not strictly necessary.

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Attempt to read with rdpcap as fallback for potential PcapReader issues
        try:
             print(f"Attempting fallback read with rdpcap for {filename}")
             packets = rdpcap(filename) # This loads the whole file into memory!
             # Re-implement the window logic using the loaded packets list
             # This part is omitted for brevity but would mirror the logic above
             print(f"Fallback read not fully implemented. Processing incomplete for {filename}.")
        except Exception as e2:
             print(f"Fallback read also failed for {filename}: {e2}")

    print(f"Finished processing {filename}. Extracted {len(features_list)} windows.")
    return features_list

def create_empty_features(window_start, window_size, label):
    """Creates a feature dictionary for an empty time window."""
    return {
        'window_start': datetime.fromtimestamp(window_start).isoformat(),
        'total_packets_sent': 0,
        'total_packets_received': 0,
        'total_packets': 0,
        'total_bytes_sent': 0,
        'total_bytes_received': 0,
        'total_bytes': 0,
        'mean_packet_size': 0,
        'median_packet_size': 0,
        'stddev_packet_size': 0,
        'variance_packet_size': 0,
        'mean_interarrival_time': 0,
        'median_interarrival_time': 0,
        'stddev_interarrival_time': 0,
        'variance_interarrival_time': 0,
        'ratio_packets_sent_received': 0,
        'ratio_bytes_sent_received': 0,
        'is_window_empty': 1,
        'low_packet_count': 1, # Also true if empty
        'label': label
    }


def extract_features(packets, window_start, window_size, label):
    """Extracts features from a list of packets within a time window."""
    
    timestamps = sorted([p['timestamp'] for p in packets])
    sizes = [p['len'] for p in packets]
    
    sent_packets = [p for p in packets if p['direction'] == 'sent']
    received_packets = [p for p in packets if p['direction'] == 'received']
    
    total_packets = len(packets)
    total_packets_sent = len(sent_packets)
    total_packets_received = len(received_packets)
    
    total_bytes = sum(sizes)
    total_bytes_sent = sum(p['len'] for p in sent_packets)
    total_bytes_received = sum(p['len'] for p in received_packets)
    
    mean_ps, median_ps, stddev_ps, variance_ps = calculate_stats(sizes)
    
    interarrival_times = []
    if len(timestamps) > 1:
        interarrival_times = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
    mean_iat, median_iat, stddev_iat, variance_iat = calculate_stats(interarrival_times)

    # Ratios - handle division by zero
    ratio_packets = 0
    if total_packets_received > 0:
        ratio_packets = total_packets_sent / total_packets_received
    elif total_packets_sent > 0:
         ratio_packets = 9999 # Indicate only sending

    ratio_bytes = 0
    if total_bytes_received > 0:
        ratio_bytes = total_bytes_sent / total_bytes_received
    elif total_bytes_sent > 0:
        ratio_bytes = 9999 # Indicate only sending
        
    # Indicator features
    is_empty = 1 if total_packets == 0 else 0 # Should not happen if called correctly, but check
    low_packet_count = 1 if total_packets <= 2 else 0

    return {
        'window_start': datetime.fromtimestamp(window_start).isoformat(),
        'total_packets_sent': total_packets_sent,
        'total_packets_received': total_packets_received,
        'total_packets': total_packets,
        'total_bytes_sent': total_bytes_sent,
        'total_bytes_received': total_bytes_received,
        'total_bytes': total_bytes,
        'mean_packet_size': mean_ps,
        'median_packet_size': median_ps,
        'stddev_packet_size': stddev_ps,
        'variance_packet_size': variance_ps,
        'mean_interarrival_time': mean_iat,
        'median_interarrival_time': median_iat,
        'stddev_interarrival_time': stddev_iat,
        'variance_interarrival_time': variance_iat,
        'ratio_packets_sent_received': ratio_packets,
        'ratio_bytes_sent_received': ratio_bytes,
        'is_window_empty': is_empty,
        'low_packet_count': low_packet_count,
        'label': label
    }

# --- Main Execution ---
if __name__ == "__main__":
    all_features = []
    for pcap_file, label in PCAP_FILES.items():
        file_features = process_pcap(pcap_file, label, TIME_WINDOW_SECONDS)
        all_features.extend(file_features)

    if not all_features:
        print("No features were extracted. Exiting.")
    else:
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Optional: Fill potential NaN values if any calculation resulted in NaN (shouldn't with current logic)
        df.fillna(0, inplace=True) 
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully processed data and saved to {OUTPUT_CSV}")
        print(f"DataFrame shape: {df.shape}")
        print("Label distribution:")
        print(df['label'].value_counts()) 