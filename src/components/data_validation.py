import sys
from src import logger
from src.utils.common import load_data
from src.entity.config_entity import DataValidationConfig
from src.exceptions import CustomException


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    
    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = load_data(self.config.local_data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

                return validation_status
            
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)
