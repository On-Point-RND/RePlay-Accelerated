"""Main module"""

import os
import logging
import warnings
import yaml
import argparse

from src_benchmarks.utils.conf import load_config, seed_everything
from src_benchmarks import TrainRunner, GridParamsSearchRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore")


def main() -> None:
    config_dir = "./src_benchmarks/configs"
    base_config_path = os.path.join(config_dir, "config.yaml")
    config = load_config(base_config_path, config_dir)
    logging.info("Configuration:\n%s", yaml.dump(config))

    seed_everything(config["env"]["SEED"])
    logging.info(f"Fixing seed: {config['env']['SEED']}")

    if config["mode"]["name"] in ["train"]:
        runner = TrainRunner(config)
    elif config["mode"]["name"] == "hyperparameter_experiment":
        runner = GridParamsSearchRunner(config)
    else:
        raise ValueError(f"Unsupported mode: {config['mode']}")

    runner.run()


if __name__ == "__main__":
    main()
