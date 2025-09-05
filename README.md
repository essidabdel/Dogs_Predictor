# 🐶 Dogs Breeds Classifier – Flask Web App  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)  
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)  
![EfficientNetV2S](https://img.shields.io/badge/Model-EfficientNetV2S-green)  

Une application **Flask** permettant d’uploader une photo de chien 🐕 et d’obtenir les **5 races les plus probables** prédites par un modèle entraîné basé sur **EfficientNetV2S** (`model.h5`).  

---

## 📂 Contenu du projet  

- `app.py` → serveur Flask + logique de prédiction  
- `templates/index.html` & `templates/result.html` → interface utilisateur (upload + résultats)  
- `static/css/style.css` → styles simples  
- `requirements.txt` → dépendances Python  
- `model.h5` → modèle EfficientNetV2S pré-entraîné + fine-tuning  
- `labels.txt` → liste des races (une par ligne, dans l’ordre du modèle)  
- `notebook.ipynb` → exploration, entraînement et tests (non nettoyé)  
- `README.md` → ce document 👋  

---

## ⚙️ Installation & exécution  

Sous **Windows PowerShell** :  

```powershell
# Créer un environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt

# Installer TensorFlow (si utilisation du modèle)
pip install tensorflow

# Lancer l’application Flask
python app.py
👉 Application accessible sur http://localhost:5000

🚀 Utilisation
Aller sur la page d’accueil

Uploader une image de chien 🐕

Le modèle EfficientNetV2S renvoie les Top 5 prédictions de races avec leurs probabilités

L’image est sauvegardée dans static/uploads/last_upload.jpg et affichée sur la page résultat

📝 Notes
Si TensorFlow n’est pas installé, un message d’erreur clair s’affiche dans l’UI

labels.txt est requis pour des sorties lisibles

EfficientNetV2S a été choisi pour son excellent compromis vitesse / précision sur la classification d’images

🎯 Exemple de résultat
📷 → Upload image

➡️ Résultats :

1️⃣ Labrador Retriever (78%)
2️⃣ Golden Retriever (12%)
3️⃣ Beagle (5%)
4️⃣ Border Collie (3%)
5️⃣ Berger Allemand (2%)

