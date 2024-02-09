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