import pandas as pd
import numpy as np
from src import logger
from src.utils.common import (load_data,
                              save_data_to_csv)
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler)
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def preprocess_data(self):
        data = load_data(self.config.source_data_path)
        logger.info(f"Total rows in data: {len(data)}")


        print("Filtering data to have only rows with purchase lead days less than 500 days")
        data =  data[data.purchase_lead < 500]
        print("Filtering data to have only rows with length of stay days less than 200 days")
        data = data[data.length_of_stay < 200]


        print("Mapping flight day data from days of the week to corresponding integer values")
        mapping = {
            "Mon" : 1,
            "Tue" : 2,
            "Wed" : 3,
            "Thu" : 4,
            "Fri" : 5,
            "Sat" : 6,
            "Sun" : 7
            }
        data.flight_day = data.flight_day.map(mapping)
        data = data.dropna(axis=0)


        print("Encoding Sales Channel and Trip Type")
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoded_data = pd.DataFrame(encoder.fit_transform(data[["sales_channel", "trip_type"]]).toarray())
        encoded_data = encoded_data.rename(columns={
            0:'internet',
            1:'mobile',
            2:'round_trip',
            3:'oneway_trip',
            4:'circle_trip'
            })
        data = data.join(encoded_data)
        print("Dropping encoded and irrelevant columns")
        categorical_columns = ['sales_channel', 'trip_type','booking_origin', 'route']
        data.drop(categorical_columns, axis=1, inplace=True)


        print("Handling missing data")
        imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
        features_columns = ['num_passengers', 'purchase_lead', "length_of_stay",
                            "flight_hour", "flight_day", "wants_extra_baggage",
                            "wants_preferred_seat", "wants_in_flight_meals", "flight_duration",
                            "internet", "mobile", "round_trip", "oneway_trip", "circle_trip"
                            ]
        imputer.fit(data[features_columns])
        data[features_columns] = imputer.fit_transform(data[features_columns])

        logger.info(f"Total rows in data after preprocessing: {len(data)}")
        data = data.dropna(axis=0)
        save_data_to_csv(data, self.config.local_data_path)

        return data
    

    def train_test_spliting(self):
        data = load_data(self.config.local_data_path)

        print("Splitting data into training and test sets")
        train, test = train_test_split(
            data, test_size=self.config.split_ratio,
            random_state=self.config.random_state)
        
        save_data_to_csv(train, self.config.train_file)
        save_data_to_csv(test, self.config.test_file)

        logger.info("Splitted data into training and test sets")
        logger.info(f"Train file shape: {train.shape}")
        logger.info(f"Train file shape: {test.shape}")

        return train, test

    
    def normalization(self):
        print("Carrying out feature scaling...")
        scaler = StandardScaler()
        train_data, test_data = self.train_test_spliting()

        train_features = train_data.drop(columns=['booking_complete'], axis=1)
        train_label = train_data.loc[:, 'booking_complete']

        test_features = test_data.drop(columns=['booking_complete'], axis=1)
        test_label = test_data.loc[:, 'booking_complete']

        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)

        train_features = pd.DataFrame(
            train_features_scaled,
            columns = train_features.columns)
        
        test_features = pd.DataFrame(
            test_features_scaled,
            columns = test_features.columns)
        
        return train_features, train_label, test_features, test_label
    

    def handling_class_imbalance(self):
        print("Handling class imbalance with SMOTEENN")
        smote_enn = SMOTEENN(random_state=self.config.random_state)
        
        data = self.normalization()
        train_features, train_label = smote_enn.fit_resample(data[0], data[1])

        save_data_to_csv(train_features, self.config.transformed_data_train_feature)
        save_data_to_csv(train_label, self.config.transformed_data_train_label)
        save_data_to_csv(data[2], self.config.transformed_data_test_feature)
        save_data_to_csv(data[3], self.config.transformed_data_test_label)

        logger.info(f"Transformed data for model training saved at {self.config.transformed_root_dir}")
    