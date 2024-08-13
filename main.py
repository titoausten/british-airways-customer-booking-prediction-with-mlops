import sys
from src import logger
from src.pipeline.stage_01_data_validation import DataValidationPipeline
from src.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from src.exceptions import CustomException


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<]\n[x==========x")
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<]\n[x==========x")
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME = "Model Training Stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    model_training = ModelTrainingPipeline()
    model_training.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<]\n[x==========x")
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<]\n[x==========x")
except Exception as e:
    raise CustomException(e, sys)
