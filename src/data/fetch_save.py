import requests
import csv
import datetime
import os
import json

def fetch_prices():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch data")
        return None

def write_to_csv(crypto_name, price):
    directory = "data/preprocessed/price"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    filename = os.path.join(directory, f"{crypto_name}.csv")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Ensure the file has headers if it's new
        if file.tell() == 0:
            writer.writerow(["timestamp", "price"])
        writer.writerow([current_time, price])

def save_raw_json(data):
    directory = "data/raw"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    filename = os.path.join(directory, "crypto_prices.json")
    
    with open(filename, mode='w') as file:
        json.dump(data, file, indent=4)


prices = fetch_prices()
if prices:
    save_raw_json(prices)
    for crypto, data in prices.items():
        write_to_csv(crypto, data["usd"])
        print(f"Logged {crypto}: {data['usd']}")


