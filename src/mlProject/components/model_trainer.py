import pandas as pd
import os
from mlProject import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path, header=0, sep=',')
        test_data = pd.read_csv(self.config.test_data_path, header=0, sep=',')

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Use RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_x, train_y.values.ravel())  # RandomForestRegressor expects 1D array for y

        # Predict on the test set
        predictions = rf.predict(test_x)

        # Evaluate the model
        rmse = mean_squared_error(test_y, predictions, squared=False)
        mae = mean_absolute_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)

        # Print or log the evaluation metrics
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2 Score:", r2)

        # Save the trained model
        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))




# import pandas as pd
# import os
# from mlProject import logger
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib
# from mlProject.entity.config_entity import ModelTrainerConfig

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
