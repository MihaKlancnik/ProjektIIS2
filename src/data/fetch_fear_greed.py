import requests
import csv
import os
import yaml
import json

def fetch_fear_greed_index(url):
    """Fetch JSON data from the Fear and Greed Index API"""
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: Status code {response.status_code}")
        return None

def parse_json_data(json_data):
    """Extract relevant fields from JSON response"""
    if not json_data or "data" not in json_data:
        print("Invalid JSON structure.")
        return []

    result = []
    for item in json_data["data"]:
        result.append({
            "date": item.get("timestamp"),
            "value": item.get("value"),
            "classification": item.get("value_classification")
        })

    return result

def write_to_csv(data_list, output_dir, filename):
    """Append new data to CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    existing_dates = set()
    file_exists = os.path.isfile(filepath)

    if file_exists:
        with open(filepath, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0] != "date":
                    existing_dates.add(row[0])

    new_data = [item for item in data_list if item["date"] not in existing_dates]
    new_data.sort(key=lambda x: x["date"])

    if not new_data:
        print("No new data to append.")
        if not file_exists:
            with open(filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["date", "value", "classification"])
        return

    with open(filepath, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["date", "value", "classification"])
        for item in new_data:
            writer.writerow([item["date"], item["value"], item["classification"]])

    print(f"Added {len(new_data)} new entries to {filepath}")
    print(f"Latest date added: {new_data[-1]['date']}")

def main():
    # Load params
    params = yaml.safe_load(open("params.yaml"))

    output_dir = params["preprocess"]["fear_greed_output_dir"]
    fear_greed_url = params["fetch"]["fear_greed_url"]
    fear_greed_csv = params["preprocess"]["fear_greed_output_filename"]

    # Fetch and save raw JSON
    raw_json = fetch_fear_greed_index(fear_greed_url)
    raw_output_path = os.path.join("data", "raw", "fear_greed_raw.json")
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    with open(raw_output_path, "w") as f:
        json.dump(raw_json, f, indent=2)

    # Parse and write processed data
    if raw_json:
        parsed_data = parse_json_data(raw_json)
        write_to_csv(parsed_data, output_dir, fear_greed_csv)

if __name__ == "__main__":
    main()
