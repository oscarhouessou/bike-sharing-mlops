import pandas as pd
import requests
import zipfile
import io
import os

def download_data(url, output_path):
    print(f"📥 Téléchargement des données depuis {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # On extrait uniquement le fichier horaire (hour.csv)
            z.extract("hour.csv", path=output_path)
        print(f"✅ Données extraites dans : {output_path}")
    else:
        print("❌ Erreur lors du téléchargement.")

if __name__ == "__main__":
    DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    RAW_DATA_DIR = "data/raw"
    
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        
    download_data(DATA_URL, RAW_DATA_DIR)