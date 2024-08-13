#import pandas as pd
from src import logger
from src.utils.common import load_data, save_bin
from sklearn.ensemble import RandomForestClassifier
from src.entity.config_entity import ModelTrainingConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    
    def train(self):
        # Load data
        train_features = load_data(self.config.train_data_features)
        train_label = load_data(self.config.train_data_label).squeeze()

        rfc = RandomForestClassifier(
            random_state = self.config.random_state,
            n_estimators= self.config.n_estimators,
            max_leaf_nodes=self.config.max_leaf_nodes,
            max_depth=self.config.max_depth)
        
        # Train model
        logger.info(f"Training started...")
        rfc.fit(train_features, train_label)
        logger.info(f"Training completed")


        # Save model
        save_bin(rfc, self.config.model_name)
