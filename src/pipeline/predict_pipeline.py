import os
import sys
import pandas as pd
from src.exceptions import CustomException
from src.utils.common import load_bin
from pydantic import BaseModel
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler)


class PredictPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = load_bin("artifacts/model_training/br_model.joblib")

        '''
        self.all_schema = {
            "num_passengers": int,
            "sales_channel": object,
            "trip_type": object,
            "purchase_lead": int,
            "length_of_stay": int,
            "flight_hour": int,
            "flight_day": object,
            "route": object,
            "booking_origin": object,
            "wants_extra_baggage": int,
            "wants_preferred_seat": int,
            "wants_in_flight_meals": int,
            "flight_duration": float
            }


    def validate_all_columns(self, data) -> bool:
        # Validate data
        try:
            validation_status = None

            all_cols = list(data.columns)
            all_schema = self.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                else:
                    validation_status = True

                return validation_status
            
        except Exception as e:
            raise CustomException(e, sys)
        '''


    def preprocess_data(self, data):
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
        categorical_columns = [
            'sales_channel',
            'trip_type',
            'booking_origin', 
            'route'
            ]
    
        data.drop(categorical_columns, axis=1, inplace=True)
        data = data.dropna(axis=0)

        # Normalization
        data_scaled = self.scaler.transform(data)
        data = pd.DataFrame(
            data_scaled,
            columns = data.columns)

        return data


    def predict(self, features):
        try:
            # Transform data
            # Load model
            # Predict
            data = self.preprocess_data(features)
            prediction = self.model.predict(data)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 num_passengers: int,
                 sales_channel: str,
                 trip_type: str,
                 purchase_lead: int,
                 length_of_stay: int,
                 flight_hour: int,
                 flight_day: str,
                 route: str,
                 booking_origin: str,
                 wants_extra_baggage: int,
                 wants_preferred_seat: int,
                 wants_in_flight_meals: int,
                 flight_duration: float):


        self.num_passengers = num_passengers
        self.sales_channel = sales_channel
        self.trip_type = trip_type
        self.purchase_lead = purchase_lead
        self.length_of_stay = length_of_stay
        self.flight_hour = flight_hour
        self.flight_day = flight_day
        self.route = route
        self.booking_origin = booking_origin
        self.wants_extra_baggage = wants_extra_baggage
        self.wants_preferred_seat = wants_preferred_seat
        self.wants_in_flight_meals = wants_in_flight_meals
        self.flight_duration = flight_duration


    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "num_passengers": [self.num_passengers],
                "sales_channel": [self.sales_channel],
                "trip_type": [self.trip_type],
                "purchase_lead": [self.purchase_lead],
                "length_of_stay": [self.length_of_stay],
                "flight_hour": [self.flight_hour],
                "flight_day": [self.flight_day],
                "route": [self.route],
                "booking_origin": [self.booking_origin],
                "wants_extra_baggage": [self.wants_extra_baggage],
                "wants_preferred_seat": [self.wants_preferred_seat],
                "wants_in_flight_meals": [self.wants_in_flight_meals],
                "flight_duration": [self.flight_duration]
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)


class FormData(BaseModel):
    num_passengers: int
    sales_channel: str
    trip_type: str
    purchase_lead: int
    length_of_stay: int
    flight_hour: int
    flight_day: str
    route: str
    booking_origin: str
    wants_extra_baggage: int
    wants_preferred_seat: int
    wants_in_flight_meals: int
    flight_duration: float
