# E2E MLOps


## Workflows

1. Update  config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the app.py


## How to run?

### Steps

Clone the repository
```bash
https://git.univ-lemans.fr/Aghilas.Sini/e2e-mlops.git
```

#### STEP 01 - Create a conda environnement  after opening the repository


```bash
conda create -n mlops_env python=3.9
```

```bash
conda activate  mlops_env
```

#### STEP 02 - install the requirements

```bash
pip install -r requirements.txt
```


```bash 
open up your local host  and port
```

## MLflow
[Documentation](https://mlflow.org/docs/latest/index.html)



#### cmd
- mlflow ui

####  dagshub
[dagshub](https://dagshub.com)


MLFLOW_TRACKING_URI=https://dagshub.com/AghilasSini/e2e-mlops.mlflow \
MLFLOW_TRACKING_USERNAME=AghilasSini \
MLFLOW_TRACKING_PASSWORD=c1f4eff3fe913cc0bdd85c69c2b98f8b5ec359ad \
python script.py

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/AghilasSini/e2e-mlops.mlflow
export MLFLOW_TRACKING_USERNAME=AghilasSini
export MLFLOW_TRACKING_PASSWORD=c1f4eff3fe913cc0bdd85c69c2b98f8b5ec359ad 
```



