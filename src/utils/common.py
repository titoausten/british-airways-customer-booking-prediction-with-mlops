import os
import sys
import pandas as pd
import yaml
import joblib
from pathlib import Path
from box import ConfigBox
from src import logger
from ensure import ensure_annotations
from src.exceptions import CustomException


@ensure_annotations
def load_data(path_to_file):
    print("Loading data file...")
    data = pd.read_csv(path_to_file, encoding="ISO-8859-1")
    logger.info(f"Data file {path_to_file} loaded successfully")
    return data


@ensure_annotations
def save_data_to_csv(data, path_to_file):
    print("Saving data file...")
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.to_csv(path_to_file, index = False)
    logger.info(f"Preprocessed data file saved at: {path_to_file}")


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")


@ensure_annotations
def save_bin(data, path: str):
    """save binary file

    Args:
        data: data to be saved as binary
        path (str): path to binary file
    """
    joblib.dump(data, open(str(path),'wb'))
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: str):
    """load binary data

    Args:
        path (str): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(open(str(path),'rb'))
    logger.info(f"binary file loaded from: {path}")
    return data
