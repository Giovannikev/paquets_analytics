# paquet_analytics

Projet d'analyse de paquets réseau pour extraire des caractéristiques, normaliser les données, projeter via PCA et détecter des anomalies.

## Prérequis
- Python 3.10+ recommandé
- Environnement virtuel (facultatif mais conseillé)

## Installation
Dans un terminal Windows :

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Dépendances optionnelles pour fonctionnalités avancées
pip install scikit-learn plotly
```

## Utilisation
Exécuter l'analyse sur un CSV (ex. paquets.csv) :

```bash
python analyze_packets.py --input paquets.csv
```

Paramètres utiles :
- `--output` dossier de sortie (par défaut: `outputs` à côté du CSV)
- `--contamination` fraction d'anomalies attendue (défaut : 0.05)
- `--top-protocols` nombre de protocoles les plus fréquents pour l'encodage (défaut : 12)

## Résultats
- `outputs/features.csv` : features + métriques de référence (euclidean_to_mean, cosine_to_mean)
- `outputs/anomalies.csv` : scores et labels d'anomalie
- `outputs/pca_projection.html` ou `.png` : visualisation 2D (PCA)

## Structure du dépôt
- `analyze_packets.py` : script principal CLI
- `requirements.txt` : dépendances minimales (numpy, pandas, matplotlib)
- `paquets.csv` : exemple de données d'entrée
- `outputs/` : exemples de résultats générés

