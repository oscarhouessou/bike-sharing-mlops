import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Chargement des données traitées
    df = pd.read_csv(config['processed_data_path'])
    X = df.drop(columns=[config['features']['target']])
    y = df[config['features']['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['train']['test_size'], random_state=config['train']['random_state']
    )

    # MLflow tracking
    mlflow.set_experiment("Bike_Sharing_Experiment")
    
    with mlflow.start_run():
        # Entraînement
        model = RandomForestRegressor(n_estimators=config['train']['n_estimators'])
        model.fit(X_train, y_train)

        # Prédictions et Métriques
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log des paramètres et métriques
        mlflow.log_params(config['train'])
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Sauvegarde du modèle (Artifact)
        os.makedirs("models", exist_ok=True)
        with open(config['train']['model_path'], "wb") as f:
            pickle.dump(model, f)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"🌲 Modèle entraîné. RMSE: {rmse:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    train("config/config.yaml")