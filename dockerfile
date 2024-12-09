# Étape 1 : Utiliser une image de base légère pour Python
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier les fichiers de votre projet dans le conteneur
COPY . /app

# Étape 4 : Installer les dépendances nécessaires
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install scikit-learn==1.3.1


# Étape 5 : Exposer le port sur lequel Flask s'exécute
EXPOSE 5000

# Étape 6 : Commande pour démarrer l'application Flask
CMD ["python", "app.py"]
