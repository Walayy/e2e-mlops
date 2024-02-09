# Application

http://149.100.158.240:2500

# Mlflow

http://149.100.158.240:2501

L'application et la partie mlflow sont hébergés sur notre serveur privé avec 2 containers docker.

# Pour build ?

## MLflow
docker build . -t mlflow -f Dockerfile_mlflow .

## Web App
docker build . -t app -f Dockerfile .

# Pour télécharger le modèle ?

https://filesender.renater.fr/?s=download&token=17c4d63d-5556-4a00-bb54-968e3f53a1be