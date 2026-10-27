
# ===== 1. Importer les bibliothèques =====
from typing import Literal  # Pour limiter les valeurs possibles d'un champ
import mlflow  # Charger le modèle depuis le Registry
import mlflow.sklearn
import pandas as pd  # Fabriquer le tableau attendu par le modèle
from fastapi import FastAPI  # Le cadre de travail de l'API
from pydantic import BaseModel  # Décrire et valider les données reçues

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")  # Le titre s'affiche dans la page /docs

# ===== 3. Charger le champion UNE SEULE FOIS, au démarrage =====
# Le recharger à chaque requête rendrait l'API très lente
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")

# ===== 4. Décrire les données attendues (validées automatiquement) =====
class Client(BaseModel):
    age: int  # Nombre entier
    salaire: float  # Nombre décimal
    abonnement: Literal["Basique", "Standard", "Premium"]  # 3 valeurs possibles
    anciennete: float
    consommation: float


# ===== 5. Route /health : vérifier que l'API est en vie =====
@app.get("/health")  # GET : on lit, on n'envoie rien
def health():
    return {"status": "ok"}


# ===== 6. Route /predict : recevoir un client, renvoyer la prédiction =====
@app.post("/predict")  # POST : on envoie un client
def predict(client: Client):  # FastAPI vérifie le JSON reçu avec Client

    # a) Transformer le client reçu en tableau d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer le texte « abonnement » en colonnes 0/1,
    # comme à l'entraînement
    ligne = pd.get_dummies(ligne)

    # c) Garder les colonnes du modèle, dans le bon ordre ;
    # les colonnes absentes valent 0
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Prédire : 1 = part, 0 = reste
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": "Le client va partir" if resultat == 1 else "Le client reste"
    }
