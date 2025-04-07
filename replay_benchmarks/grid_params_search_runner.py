import logging
import os
import json
import pickle
from pathlib import Path

import contextlib
import sys
import glob
import re

import optuna
import torch
import numpy as np
import pandas as pd

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from replay_benchmarks.utils.conf import seed_everything
from replay_benchmarks.base_runner import BaseRunner
from replay.metrics import (
    OfflineMetrics,
    Recall,
    Precision,
    MAP,
    NDCG,
    HitRate,
    MRR,
    Coverage,
    Surprisal,
)
from replay.metrics.torch_metrics_builder import metrics_to_df
from replay.models.nn.sequential import SasRec, Bert4Rec
from replay.models.nn.optimizer_utils import FatOptimizerFactory
from replay.models.nn.sequential.callbacks import (
    ValidationMetricsCallback,
    PandasPredictionCallback,
)
from replay.models.nn.sequential.postprocessors import RemoveSeenItems
from replay.models.nn.sequential.sasrec import (
    SasRecTrainingDataset,
    SasRecValidationDataset,
    SasRecPredictionDataset,
)
from replay.models.nn.sequential.bert4rec import (
    Bert4RecTrainingDataset,
    Bert4RecValidationDataset,
    Bert4RecPredictionDataset,
)


class GridParamsSearchRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.item_count = None
        self.raw_test_gt = None
        self.seq_val_dataset = None
        self.seq_test_dataset = None

        # Loggers
        self.log_dir = Path(config["paths"]["log_dir"]) / self.dataset_name / self.model_save_name
        self.csv_logger = CSVLogger(save_dir=self.log_dir / "csv_logs")
        self.tb_logger = TensorBoardLogger(save_dir=self.log_dir / "tb_logs")

        self.results_csv_path = Path(self.config["paths"]["main_csv_res_dir"]) / "results.csv"

        # self._check_paths()

    def _check_paths(self, additional_paths=None):
        """Ensure all required directories exist."""
        required_paths = [
            self.config["paths"]["log_dir"],
            self.config["paths"]["checkpoint_dir"],
            self.config["paths"]["results_dir"],
        ]
        for path in required_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

        for path in additional_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _initialize_model(self, trial=None):
        """Initialize the model based on configuration or Optuna trial parameters."""
        model_config = {
            "tensor_schema": self.tensor_schema,
        }

        if trial:
            search_space = self.config["optuna"]["search_space"][self.model_name]

            model_config.update({
                "block_count": trial.suggest_categorical("block_count", search_space["block_count"]),
                "head_count": trial.suggest_categorical("head_count", search_space["head_count"]),
                "hidden_size": trial.suggest_categorical("hidden_size", search_space["hidden_size"]),
                "max_seq_len": trial.suggest_categorical("max_seq_len", search_space["max_seq_len"]),
                "dropout_rate": trial.suggest_float("dropout_rate", float(min(search_space["dropout_rate"])), float(max(search_space["dropout_rate"])), step=0.05),
                "loss_type": trial.suggest_categorical("loss_type", search_space["loss_type"]),
            })

            optimizer_factory = FatOptimizerFactory(
                learning_rate=trial.suggest_float("learning_rate", float(min(search_space["learning_rate"])), float(max(search_space["learning_rate"])), log=True),
                weight_decay=trial.suggest_float("weight_decay", float(min(search_space["weight_decay"])), float(max(search_space["weight_decay"])), log=True),
            )
        else:
            optimizer_factory = FatOptimizerFactory(
                learning_rate=self.model_cfg["training_params"]["learning_rate"],
                weight_decay=self.model_cfg["training_params"].get("weight_decay", 0.0),
            )

        model_config.update(self.model_cfg["model_params"])

        if "sasrec" in self.model_name.lower():
            return SasRec(**model_config, optimizer_factory=optimizer_factory)
        elif "bert4rec" in self.model_name.lower():
            if self.config.get("acceleration"):
                if self.config["acceleration"].get("model"):
                    model_config.update(self.config["acceleration"]["model"])
            return Bert4Rec(**model_config, optimizer_factory=optimizer_factory)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def _prepare_dataloaders(
        self,
        seq_train_dataset,
        seq_validation_dataset,
        seq_validation_gt,
        seq_test_dataset,
    ):
        """Initialize dataloaders for training, validation, and testing."""
        logging.info("Preparing dataloaders...")

        dataset_mapping = {
            "sasrec": (
                SasRecTrainingDataset,
                SasRecValidationDataset,
                SasRecPredictionDataset,
            ),
            "bert4rec": (
                Bert4RecTrainingDataset,
                Bert4RecValidationDataset,
                Bert4RecPredictionDataset,
            ),
        }

        datasets = dataset_mapping.get(self.model_name.lower())
        if not datasets:
            raise ValueError(
                f"Unsupported model type for dataloaders: {self.model_name}"
            )

        TrainingDataset, ValidationDataset, PredictionDataset = datasets
        common_params = {
            "batch_size": self.model_cfg["training_params"]["batch_size"],
            "num_workers": self.model_cfg["training_params"]["num_workers"],
            "pin_memory": True,
        }

        train_dataloader = DataLoader(
            dataset=TrainingDataset(
                seq_train_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
            ),
            shuffle=True,
            **common_params,
        )
        val_dataloader = DataLoader(
            dataset=ValidationDataset(
                seq_validation_dataset,
                seq_validation_gt,
                seq_train_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
            ),
            **common_params,
        )
        val_pred_dataloader = DataLoader(
            dataset=PredictionDataset(
                seq_validation_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
            ),
            **common_params,
        )
        prediction_dataloader = DataLoader(
            dataset=PredictionDataset(
                seq_test_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
            ),
            **common_params,
        )

        return (
            train_dataloader,
            val_dataloader,
            val_pred_dataloader,
            prediction_dataloader,
        )

    def _load_dataloaders(self):
        """Loads data and prepares dataloaders."""
        logging.info("Preparing datasets for training.")
        train_events, validation_events, validation_gt, test_events, test_gt = (
            self.load_data()
        )
        self.validation_gt = validation_gt
        self.test_events = test_events
        self.raw_test_gt = test_gt

        (
            train_dataset,
            train_val_dataset,
            val_dataset,
            val_gt_dataset,
            test_dataset,
            test_gt_dataset,
        ) = self.prepare_datasets(
            train_events, validation_events, validation_gt, test_events, test_gt
        )
        self.item_count = train_dataset.item_count

        (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        ) = self.prepare_seq_datasets(
            train_dataset,
            train_val_dataset,
            val_dataset,
            val_gt_dataset,
            test_dataset,
            test_gt_dataset,
        )
        self.seq_val_dataset = seq_validation_dataset
        self.seq_test_dataset = seq_test_dataset

        return self._prepare_dataloaders(
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        )

    def calculate_metrics(self, predictions, ground_truth, test_events=None):
        """Calculate and return the desired metrics based on the predictions."""
        top_k = self.config["metrics"]["ks"]
        base_metrics = [
            Recall(top_k),
            Precision(top_k),
            MAP(top_k),
            NDCG(top_k),
            MRR(top_k),
            HitRate(top_k),
        ]

        diversity_metrics = []
        if test_events is not None:
            diversity_metrics = [
                Coverage(top_k),
                Surprisal(top_k),
            ]

        all_metrics = base_metrics + diversity_metrics
        metrics_results = OfflineMetrics(
            all_metrics, query_column="user_id", rating_column="score"
        )(
            predictions,
            ground_truth,
            test_events,
        )
        return metrics_to_df(metrics_results)

    def save_model(self, trainer, best_model):
        """Save the best model checkpoint to the specified directory."""
        save_path = os.path.join(
            self.config["paths"]["checkpoint_dir"],
            f"{self.model_save_name}_{self.dataset_name}",
        )
        torch.save(
            {
                "model_state_dict": best_model.state_dict(),
                "optimizer_state_dict": trainer.optimizers[0].state_dict(),
                "config": self.model_cfg,
            },
            f"{save_path}/{self.model_save_name}_checkpoint.pth",
        )

        self.tokenizer.save(f"{save_path}/sequence_tokenizer")
        self.logger.info(f"Best model saved at: {save_path}")

    def _prepare_tables_and_params(self):
        self.original_model_name = self.config['model']['save_name'] 
            
        self.batch_size_list = self.config["mode"]["batch_size"]
        self.max_seq_len = self.config["mode"]["max_seq_len"]
        self.number_launches = self.config["mode"]["number_launches"]
  
        if(self.model_cfg["model_params"]["loss_type"]=='SCE'):
            self.loss_sample_count_list = self.config["mode"]["bucket_size_y"]
        else: # if (CE or BCE or ARCFACE)
            self.loss_sample_count_list = self.config["mode"]["loss_sample_count"]

        self.array_shape = (len(self.loss_sample_count_list), 
                            len(self.batch_size_list), 
                            len(self.max_seq_len),
                            self.number_launches)
        
        self.all_metrics = self.max_allocated_memory = self.allocated_memory = -np.inf * np.ones(shape=self.array_shape)

    def _save_allocated_memory(self):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device=self.devices[0]) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device=self.devices[0]) / 1024**3  # GB
        torch.cuda.reset_peak_memory_stats()

        data = {
            'allocated_memory': [allocated],
            'max_allocated_memory': [max_allocated]
        }
        df = pd.DataFrame(data)

        df.to_csv(os.path.join(
            self.res_dir,
            "memory_stats.csv"
        ), index=False)

        self.logger.info(f"Allocated memory: {allocated} GB")
        self.logger.info(f"Max allocated memory: {max_allocated} GB")


    def _run_one_launch(self, train_dataloader, val_dataloader, val_pred_dataloader, prediction_dataloader):

        self.logger.info("Initializing model...")
        model = self._initialize_model()

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                self.config["paths"]["checkpoint_dir"],
                f"{self.model_save_name}_{self.dataset_name}",
            ),
            save_top_k=1,
            verbose=True,
            monitor="ndcg@10",
            mode="max",
        )

        early_stopping = EarlyStopping(
            monitor="ndcg@10",
            patience=self.model_cfg["training_params"]["patience"],
            mode="max",
            verbose=True,
        )

        validation_metrics_callback = ValidationMetricsCallback(
            metrics=self.config["metrics"]["types"],
            ks=self.config["metrics"]["ks"],
            item_count=self.item_count,
            postprocessors=[RemoveSeenItems(self.seq_val_dataset)],
        )

        profiler = SimpleProfiler(dirpath = self.csv_logger.log_dir, filename = 'simple_profiler')

        self.devices = [int(self.config["env"]["CUDA_VISIBLE_DEVICES"])]
        trainer = L.Trainer(
            max_epochs=self.model_cfg["training_params"]["max_epochs"],
            callbacks=[checkpoint_callback, early_stopping, validation_metrics_callback],
            logger=[self.csv_logger, self.tb_logger],
            profiler=profiler,
            precision=self.model_cfg["training_params"]["precision"],
            devices=self.devices
        )

        self.logger.info("Starting training...")
        
        trainer.fit(model, train_dataloader, val_dataloader)
        self._save_allocated_memory()

        if self.model_name.lower() == "sasrec":
            best_model = SasRec.load_from_checkpoint(checkpoint_callback.best_model_path)
        elif self.model_name.lower() == "bert4rec":
            best_model = Bert4Rec.load_from_checkpoint(checkpoint_callback.best_model_path)
        self.save_model(trainer, best_model)

        self.logger.info("Evaluating on val set...")
        pandas_prediction_callback = PandasPredictionCallback(
            top_k=max(self.config["metrics"]["ks"]),
            query_column="user_id",
            item_column="item_id",
            rating_column="score",
            postprocessors=[RemoveSeenItems(self.seq_val_dataset)],
        )
        L.Trainer(callbacks=[pandas_prediction_callback], precision=self.model_cfg["training_params"]["precision"], inference_mode=True, devices=self.devices).predict(
            best_model, dataloaders=val_pred_dataloader, return_predictions=False
        )

        result = pandas_prediction_callback.get_result()
        recommendations = self.tokenizer.query_and_item_id_encoder.inverse_transform(
            result
        )
        val_metrics = self.calculate_metrics(recommendations, self.validation_gt)
        self.logger.info(val_metrics)
        recommendations.to_parquet(
            os.path.join(
                self.res_dir,
                f"val_preds.parquet",
            ),
        )
        val_metrics.to_csv(
            os.path.join(
                self.res_dir,
                f"val_metrics.csv",
            ),
        )

        self.logger.info("Evaluating on test set...")
        pandas_prediction_callback = PandasPredictionCallback(
            top_k=max(self.config["metrics"]["ks"]),
            query_column="user_id",
            item_column="item_id",
            rating_column="score",
            postprocessors=[RemoveSeenItems(self.seq_test_dataset)],
        )
        L.Trainer(callbacks=[pandas_prediction_callback], precision=self.model_cfg["training_params"]["precision"], inference_mode=True, devices=self.devices).predict(
            best_model, dataloaders=prediction_dataloader, return_predictions=False
        )

        result = pandas_prediction_callback.get_result()
        recommendations = self.tokenizer.query_and_item_id_encoder.inverse_transform(
            result
        )
        test_metrics = self.calculate_metrics(recommendations, self.raw_test_gt, self.test_events)
        self.logger.info(test_metrics)
        recommendations.to_parquet(
            os.path.join(
                self.res_dir,
                f"test_preds.parquet",
            ),
        )
        test_metrics.to_csv(            
            os.path.join(
                self.res_dir,
                f"test_metrics.csv",
            ),
        )

    def _setup_logger(self):
        """Set up the logger to write to a file."""
        logger = logging.getLogger('ExperimentRunner')

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.log_dir / 'experiment.log')
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def generate_csv_res(self, additional_rows_dict):

        def find_latest_version_path(base_path):
            version_dirs = glob.glob(os.path.join(base_path, 'version_*'))
            version_numbers = [int(os.path.basename(d).split('_')[1]) for d in version_dirs]
            latest_version_number = max(version_numbers)
            latest_version_path = os.path.join(base_path, f'version_{latest_version_number}')
            return Path(latest_version_path)
            
        train_metrics_logs = pd.read_csv(find_latest_version_path(self.log_dir / "csv_logs/lightning_logs") / "metrics.csv")
        simple_profiler_path = find_latest_version_path(self.log_dir / "csv_logs/lightning_logs") / "fit-simple_profiler.txt"
        memory_results = pd.read_csv(self.res_dir / "memory_stats.csv")
        test_metrics = pd.read_csv(self.res_dir / "test_metrics.csv")

        combined_row = {}

        train_loss_epoch = np.round(train_metrics_logs['train_loss_epoch'].dropna().tolist(), 3)
        train_loss_step = np.round(train_metrics_logs['train_loss_step'].dropna().tolist(), 3)
        train_loss_epoch_str = ' '.join(map(str, train_loss_epoch))
        train_loss_step_str = ' '.join(map(str, train_loss_step))
        
        print(train_metrics_logs)

        print(train_loss_epoch)
        
        print(train_loss_step)

        combined_row['train_loss_epoch_val'] = train_loss_epoch_str if train_loss_epoch_str else "null"
        combined_row['train_loss_step_val'] = train_loss_step_str if train_loss_step_str else "null"

        with open(simple_profiler_path, 'r') as file:
            content = file.read()
        pattern = re.compile(r'\|  (run_training_epoch|run_training_batch)  .*?  (\d+\.\d+)')
        matches = pattern.findall(content)
        results = {}
        for match in matches:
            action, duration = match
            results[action] = float(duration)
        combined_row['run_training_epoch'] = results.get('run_training_epoch', 'Null')
        combined_row['run_training_batch'] = results.get('run_training_batch', 'Null')

        combined_row['allocated_memory'] = memory_results['allocated_memory'].iloc[0] if not memory_results['allocated_memory'].empty else 'Null'
        combined_row['max_allocated_memory'] = memory_results['max_allocated_memory'].iloc[0] if not memory_results['max_allocated_memory'].empty else 'Null'

        test_metrics.rename(columns={'Unnamed: 0': 'row_name'}, inplace=True)
        for row_index, row in test_metrics.iterrows():
            for col in test_metrics.columns:
                if col != 'row_name':
                    combined_row[f"{row['row_name']}_{col}_test"] = row[col] if not pd.isna(row[col]) else 'Null'

        for key, value in additional_rows_dict.items():
            combined_row[key] = value

        print(combined_row)

        combined_row_df = pd.DataFrame([combined_row])

        print(combined_row_df)
        print(combined_row_df.columns)

        if not pd.io.common.file_exists(self.results_csv_path):
            combined_row_df.to_csv(self.results_csv_path, index=False)
        else:
            combined_row_df.to_csv(self.results_csv_path, mode='a', header=False, index=False)


    def run(self):

        self._prepare_tables_and_params()

        for batch_size_i, batch_size in enumerate(self.batch_size_list):
            self.model_cfg["training_params"]["batch_size"] = batch_size
            
            for max_seq_len_i, max_seq_len in enumerate(self.max_seq_len): 
                self.model_cfg["model_params"]["max_seq_len"] = max_seq_len 
                
                train_dataloader, val_dataloader, val_pred_dataloader, prediction_dataloader = (
                self._load_dataloaders()
                )

                for sample_count_i, loss_sample_count in enumerate(self.loss_sample_count_list):
                    
                    for launch_number in range(self.number_launches):
                        new_seed = self.config["env"]["SEED"] + launch_number
                        seed_everything(new_seed)
                        
                        if (self.config["mode"]["loss_type"]=='SCE'):
                            n_bucket = bucket_size_x = int(2.0 * (batch_size * self.dataset_seq_len) ** 0.5)
                            self.model_cfg["model_params"]["bucket_size_x"] = bucket_size_x
                            self.model_cfg["model_params"]["bucket_size_y"] = loss_sample_count
                            self.model_cfg["model_params"]["n_buckets"] = n_bucket
                        else: # CE, BCE, ARCFACE loss function
                            self.model_cfg["model_params"]["loss_sample_count"] = loss_sample_count 
                        
                        self.model_save_name = self.model_name+f"_{batch_size=}_{loss_sample_count=}_{max_seq_len=}_{launch_number=}"
                        self.log_dir = Path(self.config["paths"]["log_dir"]) / self.dataset_name / self.model_save_name
                        self.res_dir = Path(self.config["paths"]["results_dir"]) / self.dataset_name / self.model_save_name
                        self.csv_logger = CSVLogger(save_dir=self.log_dir / "csv_logs")
                        self.tb_logger = TensorBoardLogger(save_dir=self.log_dir / "tb_logs")

                        self._check_paths([self.log_dir, self.res_dir])
                        self.logger = self._setup_logger()

                        with open(self.log_dir / 'experiment.log', 'a') as log_file:
                            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                                try:
                                    self.logger.info(f"Run experiment with:")
                                    self.logger.info(f"batch_size = {batch_size}")
                                    self.logger.info(f"loss_sample_count = {loss_sample_count}")
                                    self.logger.info(f"max_model_seq_len = {max_seq_len}")
                                    self.logger.info(f"random seed = {new_seed}")

                                    # Check if the configuration already exists in results.csv
                                    if self.results_csv_path.exists():
                                        results_df = pd.read_csv(self.results_csv_path)
                                        existing_row = results_df[
                                            (results_df['batch_size'] == batch_size) &
                                            (results_df['loss_sample_count'] == loss_sample_count) &
                                            (results_df['max_seq_len'] == max_seq_len) &
                                            (results_df['seed'] == new_seed)
                                        ]
                                        if not existing_row.empty:
                                            self.logger.info(f"Configuration already exists in results.csv. Skipping.")
                                            continue
                                
                                    ### main train launch ###
                                    self._run_one_launch(
                                                        train_dataloader=train_dataloader, 
                                                        val_dataloader=val_dataloader, 
                                                        val_pred_dataloader=val_pred_dataloader, 
                                                        prediction_dataloader=prediction_dataloader,
                                                        )
                                    #########################
                                    
                                    add_info = {
                                        'dataset': self.dataset_name,
                                        'model': self.model_name,
                                        'batch_size': batch_size,
                                        'loss_sample_count': loss_sample_count,
                                        'max_seq_len': max_seq_len,
                                        'seed': new_seed,
                                    }

                                    self.generate_csv_res(add_info)

                                # end launch_number - add line to scv
                                    
                                except RuntimeError as error_message:
                                    if str(error_message).startswith('CUDA out of memory.'):
                                        self.logger.info(f"Can not run: {batch_size=}, {loss_sample_count=}, {max_seq_len=}\n{error_message}")
                                    else:
                                        self.logger.info(f"Can not run: {batch_size=}, {loss_sample_count=}, {max_seq_len=}\n{error_message}")
                                        with open(self.log_dir / f"RuntimeError {self.model_save_name}.txt", 'w') as file:
                                            file.write(str(error_message))
                                                    
                            # end loss_sample_count
                        # end max_model_seq_len
                    # end batch_size