#!/bin/bash

model_path="/content/drive/MyDrive/TESI/Cityscapes_ema_0.9999_190000.pt"
cali_data_path="/content/drive/MyDrive/TESI/cali_data_256x32x64.pth"
log_dir="./logs"
mkdir -p $log_dir

GESCO_FLAGS="--data_dir ./data_val --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 100 --use_ddim True --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --num_classes 35 --class_cond True --no_instance False --batch_size 1 --model_path $model_path --results_path $log_dir --s 2 --one_hot_label True --snr 100 --pool None --unet_model unet"
CALIB_FLAGS="--cali_n 51 --cali_st 5 --cali_iters 20000 --cali_batch_size 1 --cali_data_path $cali_data_path"
QDIFF_FLAGS="--use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 8 --quant_mode qdiff --split --logdir $log_dir --cond"

python scripts/quantize_model.py $GESCO_FLAGS $CALIB_FLAGS $QDIFF_FLAGS
