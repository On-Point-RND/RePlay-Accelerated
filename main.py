"""Main module"""

import logging
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from replay_benchmarks import TrainRunner, InferRunner

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


@hydra.main(config_path="replay_benchmarks/configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        cfg (DictConfig): Hydra Config
    """
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    if config.mode.name == "train":
        runner = TrainRunner(config)
    elif config.mode.name == "infer":
        runner = InferRunner(config)
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    runner.run()


if __name__ == "__main__":
    main()
