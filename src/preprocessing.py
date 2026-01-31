import pandas as pd
import yaml
import os

def preprocess(config_path):
    # 1. Charger la config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Charger les données
    df = pd.read_csv(config['raw_data_path'])
    print(f"📊 Données brutes : {df.shape}")

    # 3. Feature Engineering
    # Créer une variable pour les heures de pointe (7-9h et 17-19h)
    df['is_peak'] = df['hr'].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
    
    # 4. Nettoyage
    # Supprimer les colonnes inutiles (identifiants ou fuites de données)
    df = df.drop(columns=config['features']['drop_columns'])

    # 5. Sauvegarder
    os.makedirs(os.path.dirname(config['processed_data_path']), exist_ok=True)
    df.to_csv(config['processed_data_path'], index=False)
    print(f"✅ Données traitées sauvegardées dans {config['processed_data_path']} : {df.shape}")

if __name__ == "__main__":
    preprocess("config/config.yaml")