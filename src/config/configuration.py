from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainingConfig,
                                      ModelEvaluationConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            local_data_dir=config.local_data_dir,
            STATUS_FILE=config.STATUS_FILE,
            all_schema= schema
        )

        return data_validation_config


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.split
        create_directories([config.root_dir, config.transformed_root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            source_data_path=config.source_data_path,
            local_data_path=config.local_data_path,
            transformed_root_dir=config.transformed_root_dir,
            transformed_data_train_feature=config.transformed_data_train_feature,
            transformed_data_test_feature=config.transformed_data_test_feature,
            transformed_data_train_label=config.transformed_data_train_label,
            transformed_data_test_label=config.transformed_data_test_label,
            train_file=config.train_file,
            test_file=config.test_file,
            split_ratio=params.split_ratio,
            random_state=params.random_state
        )

        return data_transformation_config
    

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        model_params = self.params.RandomForestClassifier
        params = self.params.split
        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            train_data_features=config.train_data_features,
            train_data_label=config.train_data_label,
            model_name=config.model_name,
            random_state=params.random_state,
            n_estimators=model_params.n_estimators,
            max_leaf_nodes=model_params.max_leaf_nodes,
            max_depth=model_params.max_depth
        )

        return model_training_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            train_data_features=config.train_data_features,
            train_data_label=config.train_data_label,
            test_data_features=config.test_data_features,
            test_data_label=config.test_data_features,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name,
        )

        return model_evaluation_config
