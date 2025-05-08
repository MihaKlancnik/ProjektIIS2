import requests
import csv
import os
import json
import yaml
import datetime
import io

def fetch_fear_greed_index(url):
    """Fetch Fear and Greed Index data from the API"""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch data: Status code {response.status_code}")
        return None

def parse_csv_data(csv_string):
    """Parse CSV data from the response string"""
    if not csv_string:
        return []
    
    # Parse CSV lines
    csv_reader = csv.reader(io.StringIO(csv_string))
    
    # Read all rows
    rows = list(csv_reader)
    
    if len(rows) < 2:  # Need at least header + one data row
        return []
    
    # First row is header
    header = rows[0]
    
    # Parse data rows
    result = []
    for row in rows[1:]:
        if len(row) >= 3:  # Ensure row has enough elements
            result.append({
                "date": row[0],
                "value": row[1],
                "classification": row[2]
            })
    
    return result

def write_to_csv(data_list, output_dir, filename):
    """Write processed data to CSV file, appending if file exists"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    file_exists = os.path.isfile(filepath)
    
    # Get existing dates to avoid duplicates
    existing_dates = set()
    if file_exists:
        with open(filepath, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Skip empty rows
                    existing_dates.add(row[0])  # First column is date
    
    # Open file in append mode if it exists, otherwise write mode
    mode = 'a' if file_exists else 'w'
    with open(filepath, mode=mode, newline='') as file:
        writer = csv.writer(file)
        
        # Write header only for new files
        if not file_exists:
            writer.writerow(["date", "value", "classification"])
        
        # Write new data, skipping duplicates
        new_entries = 0
        for item in data_list:
            if item["date"] not in existing_dates:
                writer.writerow([item["date"], item["value"], item["classification"]])
                new_entries += 1
    
    print(f"Added {new_entries} new entries to {filepath}")

def save_raw_data(data, raw_dir, raw_filename):
    """Save the raw CSV response"""
    os.makedirs(raw_dir, exist_ok=True)
    filepath = os.path.join(raw_dir, raw_filename)
    
    with open(filepath, mode='w') as file:
        file.write(data)
    
    print(f"Saved raw data to {filepath}")

def main():
    # Load parameters from YAML
    params = yaml.safe_load(open("params.yaml"))
    
    # Get directories and filenames from params
    output_dir = params["preprocess"]["fear_greed_output_dir"]
    raw_dir = params["preprocess"]["raw_dir"]
    fear_greed_url = params["fetch"]["fear_greed_url"]
    fear_greed_csv = params["preprocess"]["fear_greed_output_filename"]
    fear_greed_raw = params["preprocess"]["fear_greed_raw_filename"]
    
    # Fetch data
    raw_data = fetch_fear_greed_index(fear_greed_url)
    
    if raw_data:
        # Save raw data
        save_raw_data(raw_data, raw_dir, fear_greed_raw)
        
        # Parse and save processed data
        parsed_data = parse_csv_data(raw_data)
        write_to_csv(parsed_data, output_dir, fear_greed_csv)

if __name__ == "__main__":
    main()