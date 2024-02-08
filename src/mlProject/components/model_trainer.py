import pandas as pd
import numpy as np
import os
from mlProject import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path, header=0, sep=',')
        test_data = pd.read_csv(self.config.test_data_path, header=0, sep=',')

        train_x = train_data.drop([self.config.target_column], axis=1).values
        test_x = test_data.drop([self.config.target_column], axis=1).values
        train_y = train_data[[self.config.target_column]].values.ravel()
        test_y = test_data[[self.config.target_column]].values.ravel()

        # Définir les hyperparamètres à optimiser
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            # Ajoutez d'autres hyperparamètres à optimiser si nécessaire
        }

        # Initialiser RandomForestRegressor
        rf = RandomForestRegressor(random_state=42)

        # Recherche par grille pour trouver les meilleurs hyperparamètres
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(train_x, train_y)

        # Obtenez le meilleur modèle à partir de la recherche par grille
        best_rf = grid_search.best_estimator_

        # Prédire sur l'ensemble de test avec le meilleur modèle
        predictions = best_rf.predict(test_x)

        # Évaluer le modèle
        rmse = mean_squared_error(test_y, predictions, squared=False)
        mae = mean_absolute_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)

        # Afficher ou enregistrer les métriques d'évaluation
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2 Score:", r2)

        # Sauvegarder le meilleur modèle entraîné
        joblib.dump(best_rf, os.path.join(self.config.root_dir, self.config.model_name))



# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

#     def train(self):
#         train_data = pd.read_csv(self.config.train_data_path, header=0, sep=',')
#         test_data = pd.read_csv(self.config.test_data_path, header=0, sep=',')

#         train_x = train_data.drop([self.config.target_column], axis=1)
#         test_x = test_data.drop([self.config.target_column], axis=1)
#         train_y = train_data[[self.config.target_column]]
#         test_y = test_data[[self.config.target_column]]

#         # Hyperparameter tuning with GridSearchCV
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [None, 10, 20, 30],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4]
#         }

#         rf = RandomForestRegressor(random_state=42)
#         grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#         grid_search.fit(train_x, train_y.values.ravel())

#         # Use the best parameters found by GridSearchCV
#         best_rf = grid_search.best_estimator_

#         # Print or log the evaluation metrics
#         print("Best Parameters:", grid_search.best_params_)

#         # Save the trained model
#         joblib.dump(best_rf, os.path.join(self.config.root_dir, self.config.model_name))
