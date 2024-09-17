# Lightweight Diffusion Models for Resource-Constrained Semantic Communication
### [Giovanni Pignata](https://github.com/zigarov), [Eleonora Grassucci](https://sites.google.com/uniroma1.it/eleonoragrassucci/home-page), [Giordano Cicchetti](), and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)

## Abstract
Recently, generative semantic communication models have proliferated as they are revolutionizing semantic communication frameworks, improving their performance, and opening the way to novel applications. Despite their impressive ability to regenerate content from the compressed semantic information received, generative models pose crucial challenges for communication systems in terms of high memory footprints and heavy computational load. In this paper, we present a novel Quantized GEnerative Semantic COmmunication framework, Q-GESCO. The core method of Q-GESCO is a quantized semantic diffusion model capable of regenerating transmitted images from the received semantic maps while simultaneously reducing computational load and memory footprint thanks to the proposed post-training quantization technique. Q-GESCO is robust to different channel noises and obtains comparable performance to the full precision counterpart in different scenarios saving up to 75\% memory and 79\% floating point operations. This allows resource-constrained devices to exploit the generative capabilities of Q-GESCO, opening the range of applications and systems for generative semantic communication frameworks.

## Q-GESCO pipeline
<img src="/figures/qgesco_pipeline.png">

## Getting Started

### Quantize GESCO
* Train your own model or download our pretrained weights of GESCO [here](https://drive.google.com/file/d/1lW8J4gcZ3SS9r-kpEBMrVUfbC6mNLUP4/view?usp=drive_link).
* Download the Calibration Dataset (Cityscapes) [here](https://drive.google.com/file/d/1Su6rQ_ExUnNAj7srACu8v-lqdccATB-4/view?usp=sharing)
* save both files in the project directory
* Install the file `requirements.txt`
* Run the following command:

```
python scripts/quantize_model.py --data_dir ./data_val --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 100 --use_ddim True --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --num_classes 35 --class_cond True --no_instance False --batch_size 1 --model_path ./Cityscapes_ema_0.9999_190000.pt --results_path ./logs --s 2 --one_hot_label True --snr 100 --pool None --unet_model unet --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 8 --quant_mode qdiff --split --logdir ./logs --cond --cali_n 51 --cali_st 5 --cali_iters 20000 --cali_batch_size 1 --cali_data_path ./cali_data_256.pth
```

The quantized model will be saved in the corresponding log directory as `quantized_model.pth`

### Sampling with pre-quantized model
* Train your own model or download our pretrained weights of GESCO [here](https://drive.google.com/file/d/1lW8J4gcZ3SS9r-kpEBMrVUfbC6mNLUP4/view?usp=drive_link).
* Calibrate your own quantized model or download our pre-quantized model [here](https://drive.google.com/drive/folders/1ehmFaOTggBmjCTYqcYpjTnNOTcErwS1Q)
* save both models in the project directory
* Download the Dataset (Cityscapes) and save it in `./data_val` directory
* Install the file `requirements.txt`
* Run the following command:
```
python scripts/quant_sampling.py --data_dir ./data_val --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 100 --use_ddim True --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --num_classes 35 --class_cond True --no_instance False --batch_size 1 --model_path ./Cityscapes_ema_0.9999_190000.pt --results_path ./logs --s 2 --one_hot_label True --snr 100 --pool None --unet_model unet --num_samples 1 --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 8 --quant_mode qdiff --split --logdir ./logs --cond --resume --cali_ckpt ./quantized_model.pth
```

The samples will be generated in the `./logs` directory.

#### Cite
Please, cite our work if you found it useful.

```
@article{pignata2024icassp,
    title={Lightweight Diffusion Models for Resource-Constrained Semantic Communication},
    author={Pignata, Giovanni and Grassucci, Eleonora and Barbarossa, Sergio and Comminiello, Danilo},
    year={2024},
    journal={Under review at IEEE ICASSP 2025},
}
```
