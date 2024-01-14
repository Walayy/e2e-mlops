import os
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config
    

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path, header=0, sep='|')

        ## Prétraitement, on enlève les colonnes qui nous intéressent pas
        print(data)
        data = data.drop(columns=['Identifiant de document','Reference document','1 Articles CGI','2 Articles CGI', '3 Articles CGI', 
                  '4 Articles CGI', '5 Articles CGI', 'No disposition', 'Date mutation', 'No voie', 'B/T/Q', 
                  'Code voie', 'Commune', 'Code departement', 'Code commune', 'Prefixe de section', 'Section', 'No plan', 
                  'No Volume', '1er lot', 'Surface Carrez du 1er lot', '2eme lot', 'Surface Carrez du 2eme lot',
                  '3eme lot', 'Surface Carrez du 3eme lot', '4eme lot', 'Surface Carrez du 4eme lot', '5eme lot', 
                  'Surface Carrez du 5eme lot', 'Type local', 'Identifiant local', 'Nature culture', 'Nature culture speciale', 
                  'Voie', 'Type de voie'])
        print(data)

        le = LabelEncoder()
        columns_to_encode = ['Nature mutation']

        data[columns_to_encode] = data[columns_to_encode].apply(le.fit_transform)
        data = data.dropna()
        data['Valeur fonciere'] = data['Valeur fonciere'].str.replace(',', '.').astype(float)
        data['Code postal'] = data['Code postal'].astype(int)

        print(data)

        ##
        train,test =train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir,'train.csv'),index=False)
        test.to_csv(os.path.join(self.config.root_dir,'test.csv'),index=False)
        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        