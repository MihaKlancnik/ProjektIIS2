import requests
import csv
import datetime
import os
import json
import yaml

def fetch_prices(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch data")
        return None

def write_to_csv(crypto_name, price, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{crypto_name}.csv")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["timestamp", "price"])
        writer.writerow([current_time, price])

def save_raw_json(data, raw_dir, raw_filename):
    os.makedirs(raw_dir, exist_ok=True)
    filepath = os.path.join(raw_dir, raw_filename)
    with open(filepath, mode='w') as file:
        json.dump(data, file, indent=4)

def main():
    params = yaml.safe_load(open("params.yaml"))
    url = params["fetch"]["url"]
    cryptos = params["preprocess"]["cryptocurrencies"]
    output_dir = params["preprocess"]["output_dir"]
    raw_dir = params["preprocess"]["raw_dir"]
    raw_filename = params["preprocess"]["raw_filename"]

    prices = fetch_prices(url)
    if prices:
        save_raw_json(prices, raw_dir, raw_filename)
        for crypto in cryptos:
            if crypto in prices:
                write_to_csv(crypto, prices[crypto]["usd"], output_dir)
                print(f"Logged {crypto}: {prices[crypto]['usd']}")

if __name__ == "__main__":
    main()


