import requests
import csv
import os
import yaml
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

    csv_reader = csv.reader(io.StringIO(csv_string))
    rows = list(csv_reader)

    if not rows or len(rows[0]) < 3:
        return []

    # Detect header dynamically
    header = rows[0]
    header_map = {name: idx for idx, name in enumerate(header)}

    required_keys = {"fng_value", "fng_classification", "date"}
    if not required_keys.issubset(header_map.keys()):
        print("CSV header does not match expected format.")
        return []

    result = []
    for row in rows[1:]:
        if len(row) < 3:
            continue
        result.append({
            "date": row[header_map["date"]],
            "value": row[header_map["fng_value"]],
            "classification": row[header_map["fng_classification"]],
        })

    return result


def write_to_csv(data_list, output_dir, filename):
    """Write processed data to CSV file, appending if file exists"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    existing_dates = set()
    file_exists = os.path.isfile(filepath)

    if file_exists:
        with open(filepath, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header != ["date", "value", "classification"]:
                print(f"Warning: Unexpected header: {header}")
            for row in reader:
                if row and row[0] != "date":
                    existing_dates.add(row[0])

    # Sort and de-duplicate input
    new_data = [item for item in data_list if item["date"] not in existing_dates]
    new_data.sort(key=lambda x: x["date"])

    if not new_data:
        print("No new data to append.")
        # Still ensure file exists with correct header if missing
        if not os.path.exists(filepath):
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
    params = yaml.safe_load(open("params.yaml"))

    output_dir = params["preprocess"]["fear_greed_output_dir"]
    fear_greed_url = params["fetch"]["fear_greed_url"]
    fear_greed_csv = params["preprocess"]["fear_greed_output_filename"]

    raw_data = fetch_fear_greed_index(fear_greed_url)
    if raw_data:
        parsed_data = parse_csv_data(raw_data)
        write_to_csv(parsed_data, output_dir, fear_greed_csv)
    
    raw_output_path = os.path.join("data", "raw", "fear_greed_raw.csv")
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    with open(raw_output_path, "w", newline="") as f:
        f.write(raw_data)


    

if __name__ == "__main__":
    main()
