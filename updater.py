import os
from datetime import date, timedelta
import pathlib
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json

retry_strategy = Retry(
    total=4,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount('http://', adapter)
session.mount('https://', adapter)

files = []

def download_url(url, download_path, name=None):
    try:
        global files
        if name:
            file_name = os.path.join(download_path, name)
        else:
            file_name = os.path.join(download_path, os.path.basename(url))
        dir_path = os.path.dirname(file_name)
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(file_name):
            print(f"File already exists, skipping: {file_name}")
            files.append(file_name)
            return
        print(f"Attempting to download: {url}")
        response = session.get(url)
        if response.status_code == 404:
            print(f"File does not exist: {url}")
        elif response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {url} to {file_name}")
            files.append(file_name)
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

def daterange(start_date, end_date):
    days = int((end_date - start_date).days)
    print(f"Date range: {start_date} to {end_date}, {days} days")
    for n in range(days):
        yield start_date + timedelta(n)

def download_binance_daily_data(pair, training_days, region, download_path):
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    end_date = date.today() - timedelta(days=1)  # Up to yesterday
    start_date = end_date - timedelta(days=int(training_days))
    print(f"Downloading {pair} data from {start_date} to {end_date}")
    
    global files
    files = []

    with ThreadPoolExecutor() as executor:
        for single_date in daterange(start_date, end_date):
            url = f"{base_url}/{pair}/1m/{pair}-1m-{single_date}.zip"
            executor.submit(download_url, url, download_path)
    
    downloaded_files = [os.path.join(download_path, f"{pair}-1m-{d}.zip") 
                        for d in daterange(start_date, end_date) 
                        if os.path.exists(os.path.join(download_path, f"{pair}-1m-{d}.zip"))]
    print(f"Filtered {pair} files: {downloaded_files[:5]}, total: {len(downloaded_files)}")
    return downloaded_files

# Rest of the file (download_binance_current_day_data, get_coingecko_coin_id, etc.) unchanged
# ...

if __name__ == "__main__":
    download_binance_daily_data("BTCUSDT", 180, "us", "data/binance")
    download_binance_daily_data("BERAUSDT", 180, "us", "data/binance")
