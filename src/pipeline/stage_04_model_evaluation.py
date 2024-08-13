import sys
from src.exceptions import CustomException
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.components.data_transformation import DataTransformation
from src import logger


STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_dvc("train",
                                             features=model_evaluation.config.train_data_features,
                                             label=model_evaluation.config.train_data_label
                                             )
        model_evaluation.log_into_dvc("test",
                                             features=model_evaluation.config.test_data_features,
                                             label=data_transformation.config.test_file
                                             )
        model_evaluation.live.make_summary()
        model_evaluation.plot_feature_importance()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        model_evaluation = ModelEvaluationPipeline()
        model_evaluation.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
