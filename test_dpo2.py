import logging
from concordlm.config import load_config
from concordlm.trainers.dpo import run_dpo
import trl.trainer.utils

old_create = trl.trainer.utils.create_model_from_path
def intercept_create(model_id, **kwargs):
    print(f"\n[INTERCEPT] create_model_from_path called with model_id: {model_id}")
    import sys; sys.exit(0)

trl.trainer.utils.create_model_from_path = intercept_create
trl.trainer.dpo_trainer.create_model_from_path = intercept_create

logging.basicConfig(level=logging.INFO)
overrides = [
    'model.name=Qwen/Qwen2.5-0.5B-Instruct',
    'model.quantization=none',
    'model.use_flash_attention=false',
    'training.max_steps=1',
    'training.eval_strategy=no',
]
config = load_config('configs/dpo.yaml', overrides=overrides)
try:
    run_dpo(config)
except SystemExit:
    pass
