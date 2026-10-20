
import mlflow

# ===== 1. Récupérer les runs de l'expérience =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# ===== 2. Vérifier qu'il existe des résultats =====
if runs.empty:
    print("Aucun run trouvé dans l'expérience churn-prediction.")

else:
    # ===== 3. Afficher les colonnes utiles =====
    colonnes = [
        "tags.mlflow.runName",
        "params.n_estimators",
        "params.max_depth",
        "metrics.accuracy",
        "metrics.f1"
    ]

    print(runs[colonnes].to_string(index=False))

    # ===== 4. Afficher le meilleur run =====
    meilleur = runs.iloc[0]

    print()
    print(
        "Meilleur run :",
        meilleur["tags.mlflow.runName"],
        "| id :",
        meilleur["run_id"][:8],
        "..."
    )
