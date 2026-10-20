
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===== 1. Charger les données =====
df = pd.read_csv("churn_data.csv")

# ===== 2. Préparer les données =====
X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===== 3. Choisir l'expérience =====
mlflow.set_experiment("churn-prediction")

# ===== 4. Activer l'enregistrement automatique =====
mlflow.sklearn.autolog(
    serialization_format="cloudpickle"
)

# ===== 5. Entraîner le modèle =====
with mlflow.start_run(run_name="essai-auto"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Évaluer le modèle sur les données de test
    print("Score test :", model.score(X_test, y_test))
