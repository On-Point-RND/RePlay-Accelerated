"""Main module"""

import os
import logging
import warnings
import yaml
import torch

from replay_benchmarks.utils.conf import load_config, seed_everything
from replay_benchmarks import TrainRunner, InferRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore")


def main() -> None:
    # Загрузка основного конфига
    config_dir = "./replay_benchmarks/configs"
    base_config_path = os.path.join(config_dir, "config.yaml")
    config = load_config(base_config_path, config_dir)
    logging.info("Configuration:\n%s", yaml.dump(config))

    # Фиксация seed
    seed_everything(config["env"]["SEED"])
    logging.info(f"Fixing seed: {config['env']['SEED']}")

    # # Загрузка конфига модели
    # model_config_path = os.path.join(config_dir, "sasrec_megamarket.yaml")
    # model_config = load_config(model_config_path, config_dir)
    # config['model'].update(model_config['model'])

    # Проверка loss_type
    if config["model"]["params"]["model_params"]["loss_type"] == "SCE":
        # Загрузка дополнительного конфига sce_search.yaml
        sce_search_config_path = os.path.join(config_dir, "sce_search.yaml")
        sce_search_config = load_config(sce_search_config_path, config_dir)
        
        original_name =  config["model"]["save_name"]
        # Перебор гиперпараметров
        for batch_size in sce_search_config["batch_size"]:
            for bucket_size_y in sce_search_config["bucket_size_y"]:
                # Вычисление n_buckets и bucket_size_x

                try:
                    n_buckets = int((batch_size * int(config['dataset']['interactions_per_user'])) ** 0.5 * 2.)
                    bucket_size_x = int((batch_size * int(config['dataset']['interactions_per_user'])) ** 0.5 * 2.)

                    # Обновление конфига модели
                    config["model"]["params"]["training_params"]["batch_size"] = batch_size
                    config["model"]["params"]["model_params"]["n_buckets"] = n_buckets
                    config["model"]["params"]["model_params"]["bucket_size_x"] = bucket_size_x
                    config["model"]["params"]["model_params"]["bucket_size_y"] = bucket_size_y
                    config["model"]["save_name"] = original_name + f"bs={batch_size}_y={bucket_size_y}_nb={n_buckets}"
                    # Запуск обучения
                    if config["mode"]["name"] == "train":
                        runner = TrainRunner(config)
                    elif config["mode"]["name"] == "infer":
                        runner = InferRunner(config)
                    else:
                        raise ValueError(f"Unsupported mode: {config['mode']}")

                    runner.run()
                except:
                    del runner
                    torch.cuda.empty_cache()
    else:
        # Запуск обучения с текущими параметрами
        if config["mode"]["name"] == "train":
            runner = TrainRunner(config)
        elif config["mode"]["name"] == "infer":
            runner = InferRunner(config)
        else:
            raise ValueError(f"Unsupported mode: {config['mode']}")
        
        runner.run()



if __name__ == "__main__":
    main()
