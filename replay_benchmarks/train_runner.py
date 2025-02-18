import logging
import os
import yaml
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
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


class TrainRunner(BaseRunner):
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

    def run(self):
        """Execute the training pipeline."""
        
        model_save_name = self.config['model']['save_name']
        seq_len = self.config['dataset']['seq_len']      
        batch_size_list = self.model_cfg["training_params"]["batch_size"]
        number_launches = self.model_cfg["training_params"]["number_launches"]

        if (self.model_cfg["model_params"]["loss_type"]=='SCE'):
            loss_sample_count_list = self.model_cfg["model_params"]["bucket_size_y"]
        else: # CE loss function
            loss_sample_count_list = self.model_cfg["model_params"]["loss_sample_count"]
            loss_sample_count_list.append(None)
        
        all_metrics = -np.inf*np.ones(shape=(len(loss_sample_count_list), len(batch_size_list), number_launches))
        allocated_memory = -np.inf*np.ones(shape=(len(loss_sample_count_list), len(batch_size_list)))

        for batch_size_i, batch_size in enumerate(batch_size_list):
            self.model_cfg["training_params"]["batch_size"] = batch_size
            logging.info(f'batch_size = {batch_size}')

            train_dataloader, val_dataloader, prediction_dataloader = (
                self._load_dataloaders()
            )
 
            for sample_count_i, loss_sample_count in enumerate(loss_sample_count_list):  
                if (self.model_cfg["model_params"]["loss_type"]=='SCE'):
                    n_bucket = bucket_size_x = int(2.0 * (batch_size * seq_len) ** 0.5)
                    self.model_cfg["model_params"]["bucket_size_x"] = bucket_size_x
                    self.model_cfg["model_params"]["bucket_size_y"] = loss_sample_count
                    self.model_cfg["model_params"]["n_buckets"] = n_bucket
                    logging.info(f'bucket_size_y (loss_sample_count) = {loss_sample_count}')
                else: # CE loss function
                    self.model_cfg["model_params"]["loss_sample_count"] = loss_sample_count    
                    logging.info(f'loss_sample_count = {loss_sample_count}')

                self.config['model']['save_name'] = f"{model_save_name}_bs_{batch_size}_neg_{loss_sample_count}"

                try:
                    for launch_number in range(number_launches):
                        new_seed = self.config["env"]["SEED"] + launch_number
                        seed_everything(new_seed)
                        logging.info(f"Global seed: {new_seed}")
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

                        #torch.cuda.memory._record_memory_history(max_entries=100000)
                        #torch.cuda.reset_peak_memory_stats(device=None)
                        devices = [int(self.config["env"]["CUDA_VISIBLE_DEVICES"])]
                        trainer = L.Trainer(
                            max_epochs=self.model_cfg["training_params"]["max_epochs"],
                            callbacks=[checkpoint_callback, early_stopping, validation_metrics_callback],
                            logger=[self.csv_logger, self.tb_logger],
                            precision=self.model_cfg["training_params"]["precision"],
                            devices=devices
                        )

                        trainer.fit(model, train_dataloader, val_dataloader)

                        #torch.cuda.synchronize()
                        #allocated = torch.cuda.max_memory_allocated(device=None) / 1024**3 #GB
                        #torch.cuda.reset_peak_memory_stats()
                        #allocated_memory[sample_count_i, batch_size_i] = allocated
                        #pd_all_mem = pd.DataFrame(allocated_memory, columns=batch_size_list)
                        #pd_all_mem.to_csv(
                        #    os.path.join(
                        #        self.config["paths"]["log_dir"],
                        #        f"{model_save_name}_{self.dataset_name}_allocated_memory.csv",
                        #    ),
                        #)

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
                                f"{self.config['model']['save_name']}_{self.dataset_name}_test_metrics.csv",
                            ),
                        )

                        all_metrics[sample_count_i, batch_size_i, launch_number] = test_metrics['10']['NDCG']
                        np.save(os.path.join(
                                self.config["paths"]["log_dir"],
                                f"{self.model_save_name}_{self.dataset_name}_all_launches_metrics",
                            ), all_metrics)
                    # end for launch_number in range(number_launches):
                except RuntimeError as error_message:
                    print(error_message)
                    logging.info(f"Can not run batch_size = {batch_size} and negative_samples = {loss_sample_count}")
            # end for sample_count_i, loss_sample_count in enumerate(loss_sample_count_list):
        # end for batch_size_i, batch_size in enumerate(batch_size_list):
        
        mean_metric = all_metrics.mean(axis=-1)
        std_metric = all_metrics.std(axis=-1)

        confidence = 0.95
        if(number_launches>1):
            confidence = 0.95
            t_value = st.t.ppf((1 + confidence) / 2, df=number_launches-1)
            margin_of_error = t_value * (std_metric / np.sqrt(number_launches))
            lower_bound = mean_metric - margin_of_error
            upper_bound = mean_metric + margin_of_error
        else:
            lower_bound, upper_bound = mean_metric, mean_metric
        
        plt.figure(figsize=(12,10))
        for bs_i, bs in enumerate(batch_size_list): 
            if(self.model_cfg["model_params"]["loss_type"]=='SCE'): 
                lscl = loss_sample_count_list
                mm = mean_metric[:, bs_i]
                ub = upper_bound[:, bs_i]
                lb = lower_bound[:, bs_i]
            else:
                lscl = loss_sample_count_list[:-1]
                mm = mean_metric[:-1, bs_i]
                ub = upper_bound[:-1, bs_i]
                lb = lower_bound[:-1, bs_i]
                # if pure CE
                if((mean_metric[-1, bs_i]>0).all()):
                    metric_full = mean_metric[-1, bs_i]    
                    plt.plot([lscl[0], lscl[-1]], [metric_full, metric_full], '--', label=f'{self.model_cfg["model_params"]["loss_type"]}_all_negative_bs={bs}')
                # end if pure CE

            plt.plot(lscl, mm, 'o-', label=f'{self.model_cfg["model_params"]["loss_type"]}_metrics_bs={bs}')
            plt.fill_between(lscl, lb, ub, color='b', alpha=0.2)
            plt.xscale('log')

            for xi, yi in zip(lscl, mm):
                plt.plot([xi, xi],[yi, all_metrics[all_metrics>0].min()], 'b--', alpha=0.25)
        for xi in lscl:
            plt.text(xi, all_metrics[all_metrics>0].min(), f"{xi:.1f}", fontsize=10, ha='right')
            
        plt.legend()
        plt.title(f'Metrics with {self.model_cfg["model_params"]["loss_type"]}')
        plt.xlabel('Number samples')
        plt.ylabel('NDCG@10 metric')
        plt.grid('on')   
        path_for_save = os.path.join(
            self.config["paths"]["results_dir"],
            f"{self.model_save_name}_{self.dataset_name}_metrics_convergence.png",
            )
        plt.savefig(path_for_save)
