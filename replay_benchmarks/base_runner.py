import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from omegaconf import DictConfig

from replay.data import (
    FeatureHint,
    FeatureInfo,
    FeatureSchema,
    FeatureSource,
    FeatureType,
    Dataset,
)
from replay.splitters import LastNSplitter
from replay.utils import DataFrameLike
from replay.data.nn import (
    SequenceTokenizer,
    SequentialDataset,
    TensorFeatureSource,
    TensorSchema,
    TensorFeatureInfo,
)


class BaseRunner(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.model_name = config.model.name
        self.dataset_cfg = config.dataset
        self.model_cfg = config.model
        self.mode = config.mode.name
        self.tokenizer = None
        self.interactions = None
        self.user_features = None
        self.item_features = None
        self.tensor_schema = self.build_tensor_schema()
        self.setup_environment()

    def load_data(
        self,
    ) -> Tuple[
        DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike
    ]:
        """Load dataset and split into train, validation, and test sets."""
        self.interactions = pd.read_csv(os.path.join(self.dataset_cfg.path, 'interactions.csv'))
        self.user_features = pd.read_csv(os.path.join(self.dataset_cfg.path, 'user_features.csv'))
        self.item_features = pd.read_csv(os.path.join(self.dataset_cfg.path, 'item_features.csv'))
        splitter = LastNSplitter(
            N=1,
            divide_column=self.dataset_cfg.feature_schema.query_column,
            query_column=self.dataset_cfg.feature_schema.query_column,
            strategy="interactions",
        )

        train_events, validation_events, validation_gt, test_events, test_gt = (
            self._split_data(splitter, self.interactions)
        )
        logging.info("Data split into train, validation, and test sets")
        return train_events, validation_events, validation_gt, test_events, test_gt

    def _split_data(
        self, splitter: LastNSplitter, interactions: pd.DataFrame
    ) -> Tuple[
        DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike, DataFrameLike
    ]:
        """Split data for training, validation, and testing."""
        test_events, test_gt = splitter.split(interactions)
        validation_events, validation_gt = splitter.split(test_events)
        train_events = validation_events
        return train_events, validation_events, validation_gt, test_events, test_gt

    def prepare_feature_schema(self, is_ground_truth: bool) -> FeatureSchema:
        """Prepare the feature schema based on whether ground truth is needed."""
        base_features = FeatureSchema(
            [
                FeatureInfo(
                    column=self.dataset_cfg.feature_schema.query_column,
                    feature_hint=FeatureHint.QUERY_ID,
                    feature_type=FeatureType.CATEGORICAL,
                ),
                FeatureInfo(
                    column=self.dataset_cfg.feature_schema.item_column,
                    feature_hint=FeatureHint.ITEM_ID,
                    feature_type=FeatureType.CATEGORICAL,
                ),
            ]
        )
        if is_ground_truth:
            return base_features

        return base_features + FeatureSchema(
            [
                FeatureInfo(
                    column=self.dataset_cfg.feature_schema.timestamp_column,
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.TIMESTAMP,
                ),
            ]
        )

    def build_tensor_schema(self) -> TensorSchema:
        """Build TensorSchema for the sequential model."""
        embedding_dim = self.model_cfg.params.embedding_dim
        item_feature_name = "item_id_seq"

        return TensorSchema(
            TensorFeatureInfo(
                name=item_feature_name,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[
                    TensorFeatureSource(
                        FeatureSource.INTERACTIONS,
                        self.dataset_cfg.feature_schema.item_column,
                    )
                ],
                feature_hint=FeatureHint.ITEM_ID,
                embedding_dim=embedding_dim,
            )
        )

    def prepare_datasets(
        self,
        train_events: DataFrameLike,
        validation_events: DataFrameLike,
        validation_gt: DataFrameLike,
        test_events: DataFrameLike,
        test_gt: DataFrameLike,
    ) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset]:
        """Prepare Dataset objects for training, validation, and testing."""
        feature_schema = self.prepare_feature_schema(is_ground_truth=False)
        ground_truth_schema = self.prepare_feature_schema(is_ground_truth=True)

        train_dataset = Dataset(
            feature_schema=feature_schema,
            interactions=train_events,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        validation_dataset = Dataset(
            feature_schema=feature_schema, 
            interactions=validation_events,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        validation_gt_dataset = Dataset(
            feature_schema=ground_truth_schema,
            interactions=validation_gt,
            check_consistency=True,
            categorical_encoded=False,
        )
        test_dataset = Dataset(
            feature_schema=feature_schema,
            interactions=test_events,
            query_features=self.user_features,
            item_features=self.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        test_gt_dataset = Dataset(
            feature_schema=ground_truth_schema,
            interactions=test_gt,
            check_consistency=True,
            categorical_encoded=False,
        )

        return (
            train_dataset,
            validation_dataset,
            validation_gt_dataset,
            test_dataset,
            test_gt_dataset,
        )

    def prepare_seq_datasets(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        validation_gt: Dataset,
        test_dataset: Dataset,
        test_gt: Dataset,
    ) -> Tuple[
        SequentialDataset, SequentialDataset, SequentialDataset, SequentialDataset
    ]:
        """Prepare SequentialDataset objects for training, validation, and testing."""
        self.tokenizer = self.tokenizer or self._initialize_tokenizer(train_dataset)

        seq_train_dataset = self.tokenizer.transform(train_dataset)
        seq_validation_dataset, seq_validation_gt = self._prepare_sequential_validation(
            validation_dataset, validation_gt
        )
        seq_test_dataset = self._prepare_sequential_test(test_dataset, test_gt)

        return (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        )

    def _initialize_tokenizer(self, train_dataset: Dataset) -> SequenceTokenizer:
        """Initialize and fit the SequenceTokenizer."""
        tokenizer = SequenceTokenizer(self.tensor_schema, allow_collect_to_master=True)
        tokenizer.fit(train_dataset)
        return tokenizer

    def _prepare_sequential_validation(
        self, validation_dataset: Dataset, validation_gt: Dataset
    ) -> Tuple[SequentialDataset, SequentialDataset]:
        """Prepare sequential datasets for validation."""
        seq_validation_dataset = self.tokenizer.transform(validation_dataset)
        seq_validation_gt = self.tokenizer.transform(
            validation_gt, [self.tensor_schema.item_id_feature_name]
        )

        return SequentialDataset.keep_common_query_ids(
            seq_validation_dataset, seq_validation_gt
        )

    def _prepare_sequential_test(
        self, test_dataset: Dataset, test_gt: Dataset
    ) -> SequentialDataset:
        """Prepare sequential dataset for testing."""
        test_query_ids = test_gt.query_ids
        test_query_ids_np = self.tokenizer.query_id_encoder.transform(test_query_ids)[
            "user_id"
        ].values
        return self.tokenizer.transform(test_dataset).filter_by_query_id(
            test_query_ids_np
        )

    def setup_environment(self):
        os.environ["CUDA_DEVICE_ORDER"] = self.config.env.CUDA_DEVICE_ORDER
        os.environ["OMP_NUM_THREADS"] = self.config.env.OMP_NUM_THREADS
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.env.CUDA_VISIBLE_DEVICES

    @abstractmethod
    def run(self, config: DictConfig):
        """Run method to be implemented in derived classes.

        Args:
            config (DictConfig): Hydra Config with the parameters.
        """
        raise NotImplementedError
