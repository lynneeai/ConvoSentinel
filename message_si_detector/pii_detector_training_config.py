import json
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Training_Config:
    output_dir: str
    
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    num_train_epochs: int = 20
    learning_rate: float = 5e-5
    
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    logging_steps: int = 10
    report_to: str = "tensorboard"
    logging_dir: str = None
    
    save_strategy: str = "steps"
    save_steps: str = 100
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    fp16: bool = False
    
    freeze_backbone: bool = False
    eval_size: float = 0.2
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}
    
    def to_json_string(self):
        return json.dumps(self.to_dict())