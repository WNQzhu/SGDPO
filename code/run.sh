export PATH=/opt/conda/bin:$PATH


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/mistral-7b-base-simpo.yaml
