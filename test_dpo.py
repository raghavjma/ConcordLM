import logging
from concordlm.config import load_config
logging.basicConfig(level=logging.INFO)
overrides = [
    'model.name=Qwen/Qwen2.5-0.5B-Instruct',
    'model.quantization=none',
    'model.use_flash_attention=false',
    'training.max_steps=1',
    'training.eval_strategy=no',
]
config = load_config('configs/dpo.yaml', overrides=overrides)

from concordlm.trainers.dpo import run_dpo
import trl
class FakeDPOTrainer:
    def __init__(self, model=None, ref_model=None, args=None, **kwargs):
        print(f"FakeDPOTrainer initialized!")
        print(f"model = {model}")
        print(f"ref_model = {ref_model}")
        if args and hasattr(args, "model_init_kwargs"):
            print(f"model_init_kwargs = {args.model_init_kwargs}")
        import sys; sys.exit(0)

trl.DPOTrainer = FakeDPOTrainer
try:
    run_dpo(config)
except SystemExit:
    pass
