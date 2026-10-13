import pandas as pd  # Pandas : pour construire et manipuler notre tableau de données
import numpy as np   # NumPy : pour générer des valeurs numériques aléatoires

# On fixe une graine aléatoire (seed).
# Cela permet de reproduire exactement les mêmes données
# à chaque exécution du programme.
np.random.seed(42)

# Nombre de clients fictifs à générer
n = 500

# On construit le tableau colonne par colonne.
# Chaque colonne ci-dessous représente une FEATURE
# (donnée d'entrée du futur modèle).
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),  
    # Âge compris entre 18 et 69 ans

    "salaire": np.random.randint(1200, 8000, n),  
    # Salaire mensuel compris entre 1200 et 7999

    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"], n
    ),  
    # Type d'offre du client

    "anciennete": np.random.randint(0, 15, n),  
    # Ancienneté comprise entre 0 et 14 ans

    "consommation": np.random.randint(0, 100, n),  
    # Score d'utilisation du service compris entre 0 et 99
})

# Création de la TARGET (variable à prédire) : churn.
#
# Règle simplifiée :
# Un client est considéré comme étant à risque de départ si :
# - sa consommation est inférieure à 30
# ET
# - son ancienneté est inférieure à 2 ans.
#
# Le résultat True/False est converti en 1/0.
data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)

# Sauvegarde du tableau dans un fichier CSV
data.to_csv("churn_data.csv", index=False)

# Affichage des premières lignes pour vérifier les données
print(data.head())

# Calcul et affichage du taux de churn
print(f"Taux de churn : {data['churn'].mean():.1%}")