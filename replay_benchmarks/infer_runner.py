from omegaconf import DictConfig

from replay.models.nn.sequential import SasRec, Bert4Rec

from replay.utils import get_spark_session
from .base_runner import BaseRunner


class InferRunner(BaseRunner):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.spark_session = get_spark_session()
        self.model = self.load_model()

    def load_model(self):
        """Load the appropriate model based on the config."""
        if self.model_name.lower() == "sasrec":
            model = SasRec(
                tensor_schema=self.model_cfg.tensor_schema,
                block_count=self.model_cfg.block_count,
                head_count=self.model_cfg.head_count,
                max_seq_len=self.model_cfg.max_seq_len,
                hidden_size=self.model_cfg.hidden_size,
                dropout_rate=self.model_cfg.dropout_rate,
            )
        elif self.model_name.lower() == "bert4rec":
            model = Bert4Rec(
                tensor_schema=self.model_cfg.tensor_schema,
                block_count=self.model_cfg.block_count,
                head_count=self.model_cfg.head_count,
                max_seq_len=self.model_cfg.max_seq_len,
                hidden_size=self.model_cfg.hidden_size,
                dropout_rate=self.model_cfg.dropout_rate,
            )
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

        model = model.load_from_checkpoint(self.config.artifacts.checkpoint_dir)
        model.eval()
        return model

    def run(self):
        """Main inference logic."""
        pass
