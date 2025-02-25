import logging
import os
import yaml
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from replay_benchmarks.utils.conf import seed_everything

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

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


class ExperimentRunner(BaseRunner):
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

        self._check_paths()

    def _check_paths(self):
        """Ensure all required directories exist."""
        required_paths = [
            self.config["paths"]["log_dir"],
            self.config["paths"]["checkpoint_dir"],
            self.config["paths"]["results_dir"],
        ]
        for path in required_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _initialize_model(self):
        """Initialize the model based on the configuration."""
        model_config = {
            "tensor_schema": self.tensor_schema,
            "optimizer_factory": FatOptimizerFactory(
                learning_rate=self.model_cfg["training_params"]["learning_rate"]
            ),
        }
        model_config.update(self.model_cfg["model_params"])

        if "sasrec" in self.model_name.lower():
            return SasRec(**model_config)
        elif "bert4rec" in self.model_name.lower():
            if self.config.get("acceleration"):
                if self.config["acceleration"].get("model"):
                    model_config.update(self.config["acceleration"]["model"])
            return Bert4Rec(**model_config)
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
        prediction_dataloader = DataLoader(
            dataset=PredictionDataset(
                seq_test_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
            ),
            **common_params,
        )

        return train_dataloader, val_dataloader, prediction_dataloader

    def _load_dataloaders(self):
        """Loads data and prepares dataloaders."""
        logging.info("Preparing datasets for training.")
        train_events, validation_events, validation_gt, test_events, test_gt = (
            self.load_data()
        )
        self.raw_test_gt = test_gt

        train_dataset, val_dataset, val_gt_dataset, test_dataset, test_gt_dataset = (
            self.prepare_datasets(
                train_events, validation_events, validation_gt, test_events, test_gt
            )
        )
        self.item_count = train_dataset.item_count

        (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        ) = self.prepare_seq_datasets(
            train_dataset, val_dataset, val_gt_dataset, test_dataset, test_gt_dataset
        )
        self.seq_val_dataset = seq_validation_dataset
        self.seq_test_dataset = seq_test_dataset

        return self._prepare_dataloaders(
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        )

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate and return the desired metrics based on the predictions."""
        top_k = self.config["metrics"]["ks"]
        metrics_list = [
            Recall(top_k),
            Precision(top_k),
            MAP(top_k),
            NDCG(top_k),
            MRR(top_k),
            HitRate(top_k),
        ]
        metrics = OfflineMetrics(
            metrics_list, query_column="user_id", rating_column="score"
        )(predictions, ground_truth)
        return metrics_to_df(metrics)

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
        logging.info(f"Best model saved at: {save_path}")

    def _prepare_tables_and_params(self):
        self.original_model_name = self.config['model']['save_name'] 
            
        self.batch_size_list = self.config["mode"]["batch_size"]
        self.max_seq_len = self.config["mode"]["max_seq_len"]
        self.number_launches = self.config["mode"]["number_launches"]

        self.dataset_seq_len = self.config['dataset']['seq_len'] 
  
        if(self.model_cfg["model_params"]["loss_type"]=='SCE'):
            self.loss_sample_count_list = self.config["mode"]["bucket_size_y"]
        else: # if (CE or BCE or ARCFACE)
            self.loss_sample_count_list = self.config["mode"]["loss_sample_count"]

        self.array_shape = (len(self.loss_sample_count_list), 
                            len(self.batch_size_list), 
                            len(self.max_seq_len),
                            self.number_launches)
        
        self.all_metrics = self.max_allocated_memory = self.allocated_memory = -np.inf * np.ones(shape=self.array_shape)
    
    def _save_allocated_memory(self, indeces):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device=self.devices[0]) / 1024**3 #GB
        max_allocated = torch.cuda.max_memory_allocated(device=self.devices[0]) / 1024**3 #GB     
        torch.cuda.reset_peak_memory_stats()
        self.allocated_memory[indeces] = allocated
        self.max_allocated_memory[indeces] = max_allocated

        np.save(os.path.join(
                self.config["paths"]["log_dir"],
                self.dataset_name,
                f"{self.model_save_name}_allocated_memory",
            ), self.allocated_memory)
        np.save(os.path.join(
                self.config["paths"]["log_dir"],
                self.dataset_name,
                f"{self.model_save_name}_max_allocated_memory",
            ), self.max_allocated_memory)
    
    def _save_all_launches_metrics(self, indeces, test_metric):
        self.all_metrics[indeces] = test_metric #test_metrics['10']['NDCG']
        np.save(os.path.join(
                self.config["paths"]["log_dir"],
                self.dataset_name,
                f"{self.model_save_name}_all_launches_metrics",
            ), self.all_metrics)
    
    def _run_one_launch(self, indeces, train_dataloader, val_dataloader, prediction_dataloader):
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

        self.devices = [int(self.config["env"]["CUDA_VISIBLE_DEVICES"])]
        trainer = L.Trainer(
            max_epochs=self.model_cfg["training_params"]["max_epochs"],
            callbacks=[checkpoint_callback, early_stopping, validation_metrics_callback],
            logger=[self.csv_logger, self.tb_logger],
            precision=self.model_cfg["training_params"]["precision"],
            devices=self.devices
        )

        trainer.fit(model, train_dataloader, val_dataloader)
        self._save_allocated_memory(indeces)

        if self.model_name.lower() == "sasrec":
            best_model = SasRec.load_from_checkpoint(checkpoint_callback.best_model_path)
        elif self.model_name.lower() == "bert4rec":
            best_model = Bert4Rec.load_from_checkpoint(checkpoint_callback.best_model_path)
        self.save_model(trainer, best_model)

        logging.info("Evaluating on test set...")
        pandas_prediction_callback = PandasPredictionCallback(
            top_k=max(self.config["metrics"]["ks"]),
            query_column="user_id",
            item_column="item_id",
            rating_column="score",
            postprocessors=[RemoveSeenItems(self.seq_test_dataset)],
        )
        L.Trainer(callbacks=[pandas_prediction_callback], inference_mode=True, devices=self.devices).predict(
            best_model, dataloaders=prediction_dataloader, return_predictions=False
        )

        result = pandas_prediction_callback.get_result()
        recommendations = self.tokenizer.query_and_item_id_encoder.inverse_transform(
            result
        )
        test_metrics = self.calculate_metrics(recommendations, self.raw_test_gt)
        logging.info(test_metrics)
        test_metrics.to_csv(
            os.path.join(
                self.config["paths"]["results_dir"],
                f"{self.config['model']['save_name']}_{self.dataset_name}_test_metrics.csv",
            ),
        )

        self._save_all_launches_metrics(indeces, test_metrics['10']['NDCG'])
    

    def run(self):

        self._prepare_tables_and_params()

        for batch_size_i, batch_size in enumerate(self.batch_size_list):
            self.model_cfg["training_params"]["batch_size"] = batch_size
            
            for max_seq_len_i, max_seq_len in enumerate(self.max_seq_len): 
                self.model_cfg["model_params"]["max_seq_len"] = max_seq_len 
                
                train_dataloader, val_dataloader, prediction_dataloader = (
                    self._load_dataloaders()
                )

                for sample_count_i, loss_sample_count in enumerate(self.loss_sample_count_list):
                    if (self.config["mode"]["loss_type"]=='SCE'):
                        n_bucket = bucket_size_x = int(2.0 * (batch_size * self.dataset_seq_len) ** 0.5)
                        self.model_cfg["model_params"]["bucket_size_x"] = bucket_size_x
                        self.model_cfg["model_params"]["bucket_size_y"] = loss_sample_count
                        self.model_cfg["model_params"]["n_buckets"] = n_bucket
                    else: # CE, BCE, ARCFACE loss function
                        self.model_cfg["model_params"]["loss_sample_count"] = loss_sample_count 
                    
                    try:
                        for launch_number in range(self.number_launches):
                            new_seed = self.config["env"]["SEED"] + launch_number
                            seed_everything(new_seed)

                            logging.info(f"Run experiment with:")
                            logging.info(f"batch_size = {batch_size}")
                            logging.info(f"loss_sample_count = {loss_sample_count}")
                            logging.info(f"max_model_seq_len = {max_seq_len}")
                            logging.info(f"random seed = {new_seed}")

                        
                            ### main train launch ###
                            self._run_one_launch(indeces=(batch_size_i, sample_count_i, max_seq_len_i),
                                                    train_dataloader=train_dataloader, 
                                                    val_dataloader=val_dataloader,
                                                    prediction_dataloader=prediction_dataloader)
                            #########################

                        # end launch_number
                    except RuntimeError as error_message:
                        if str(error_message).startswith('CUDA out of memory.'):
                            logging.info(f"Can not run: {batch_size=}, {loss_sample_count=}, {max_seq_len=}")
                        else:
                            error_path = os.path.join(
                                self.config["paths"]["log_dir"],
                                self.dataset_name,
                                f"RuntimeError {self.model_save_name}.txt",
                            )
                            with open(error_path, 'w') as file: 
                                file.write(str(error_message)) 
                                          
                # end loss_sample_count
            # end max_model_seq_len
        # end batch_size

        logging.info("Initializing model...")
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

        devices = [int(self.config["env"]["CUDA_VISIBLE_DEVICES"])]
        trainer = L.Trainer(
            max_epochs=self.model_cfg["training_params"]["max_epochs"],
            callbacks=[checkpoint_callback, early_stopping, validation_metrics_callback],
            logger=[self.csv_logger, self.tb_logger],
            precision=self.model_cfg["training_params"]["precision"],
            devices=devices
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        if self.model_name.lower() == "sasrec":
            best_model = SasRec.load_from_checkpoint(checkpoint_callback.best_model_path)
        elif self.model_name.lower() == "bert4rec":
            best_model = Bert4Rec.load_from_checkpoint(checkpoint_callback.best_model_path)
        self.save_model(trainer, best_model)

        logging.info("Evaluating on test set...")
        pandas_prediction_callback = PandasPredictionCallback(
            top_k=max(self.config["metrics"]["ks"]),
            query_column="user_id",
            item_column="item_id",
            rating_column="score",
            postprocessors=[RemoveSeenItems(self.seq_test_dataset)],
        )
        L.Trainer(callbacks=[pandas_prediction_callback], inference_mode=True, devices=devices).predict(
            best_model, dataloaders=prediction_dataloader, return_predictions=False
        )

        result = pandas_prediction_callback.get_result()
        recommendations = self.tokenizer.query_and_item_id_encoder.inverse_transform(
            result
        )
        test_metrics = self.calculate_metrics(recommendations, self.raw_test_gt)
        logging.info(test_metrics)
        test_metrics.to_csv(
            os.path.join(
                self.config["paths"]["results_dir"],
                f"{self.model_save_name}_{self.dataset_name}_test_metrics.csv",
            ),
        )

