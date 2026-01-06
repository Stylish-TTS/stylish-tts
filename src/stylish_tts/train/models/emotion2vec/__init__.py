# Avoid installing OmegaConf and FunASR
from .model import Emotion2vec as _Emotion2Vec
from huggingface_hub import snapshot_download
from munch import Munch
from pathlib import Path
import yaml
import torch


class Emotion2Vec(_Emotion2Vec):
    @classmethod
    def from_pretrained(cls, hf_repo_id="emotion2vec/emotion2vec_base"):
        model_dir = Path(snapshot_download(hf_repo_id))
        config_path = model_dir / "config.yaml"
        pt_path = next(model_dir.glob("*.pt"))
        with config_path.open("r", encoding="utf-8") as file:
            cfg = Munch.fromDict(yaml.safe_load(file))
            # Stupid bug: https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
            cfg.model_conf.norm_eps = float(cfg.model_conf.norm_eps)
        model = cls(**cfg)
        model.load_state_dict(torch.load(pt_path, map_location="cpu")["model"])
        return model
