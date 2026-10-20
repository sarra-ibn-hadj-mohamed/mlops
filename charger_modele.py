
import mlflow
import mlflow.sklearn
import pandas as pd

# ===== 1. Retrouver le meilleur run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

if runs.empty or pd.isna(runs.iloc[0]["metrics.f1"]):
    raise ValueError("Aucun run avec un F1-score disponible.")

run_id = runs.iloc[0]["run_id"]

# ===== 2. Recharger le modèle =====
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Modèle rechargé :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)

# ===== 3. Créer un nouveau client =====
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# ===== 4. Adapter les colonnes au modèle =====
if not hasattr(model, "feature_names_in_"):
    raise ValueError("Les noms des variables ne sont pas disponibles.")

colonnes = list(model.feature_names_in_)

for colonne in colonnes:
    if colonne not in client.columns:
        client[colonne] = 0

client = client[colonnes]

# ===== 5. Prédire le départ du client =====
prediction = model.predict(client)[0]

print(
    "Le client va-t-il partir ? ->",
    "OUI" if prediction == 1 else "NON"
)
