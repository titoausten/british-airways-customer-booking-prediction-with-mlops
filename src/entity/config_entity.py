from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    local_data_dir: Path
    STATUS_FILE: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    source_data_path: Path
    local_data_path: Path
    transformed_root_dir: Path
    transformed_data_train_feature: Path
    transformed_data_test_feature: Path
    transformed_data_train_label: Path
    transformed_data_test_label: Path
    train_file: Path
    test_file: Path
    split_ratio: float
    random_state: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_features: Path
    train_data_label: Path
    model_name: str
    random_state: int
    n_estimators: int
    max_leaf_nodes: int
    max_depth: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    train_data_features: Path
    train_data_label: Path
    test_data_features: Path
    test_data_label: Path
    model_path: str
    metric_file_name: Path
