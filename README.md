# Projet d'expérimentation AI

Package manager uv

libs utilisées :
 - **jupyterlab ipykernel ipywidgets** gestion notebooks
 - **torch transformers accelerate sentencepiece** chargement des modeles
 - **bitsandbytes** quantification modele
 - **huggingface_hub** chargement modele depuis HF\n
Chargement d'un modele "mistralai/Mistral-7B-Instruct-v0.2" depuis HF (llm/model.py)\n
Le script app_console.py permet de lancer une boucle de chat avec historique de conversation dans la console\n
Jupyter lab permet de tester des blocs de scripts

## Démarrage du projet
Lancer le kernel jupyter\n
```uv run python -m ipykernel install --user --name ai-learning --display-name "AI Learning (RTX 5090)"```

2 méthodes de lancement :

- Lancer jupyter lab et exécuter le notebook Mistral-test.ipynb\n
```uv run jupyter lab```
- Lancer le script console\n
```uv run python scripts/app_console.py```
