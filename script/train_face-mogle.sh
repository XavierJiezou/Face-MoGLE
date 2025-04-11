# Specify the config file path and the GPU devices to use
export CUDA_VISIBLE_DEVICES=0,1

# Specify the config file path
export XFL_CONFIG=config/Face-MoGLE.yaml


echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41355 -m src.train.train