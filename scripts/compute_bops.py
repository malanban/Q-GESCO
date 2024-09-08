"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import torch
# import torch.distributed as dist
import torch.autograd.profiler as profiler
# import torchvision as tv
# import torchprofile
# from torchprofile import profile_macs

from guided_diffusion.image_datasets import load_data

# from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import numpy as np
import matplotlib.pyplot as plt
import os
# from scipy import ndimage, signal
from pooling import MedianPool2d
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# SNR (var): 1 (0.9) 5 (0.6) 10 (0.36) 15 (0.22) 20 (0.13) 25 (0.08) 30 (0.05) 100 (0.0)
SNR_DICT = {100: 0.0,
            30: 0.05,
            25: 0.08,
            20: 0.13,
            15: 0.22,
            10: 0.36,
            5: 0.6,
            1: 0.9}

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    # logger.configure()
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # Carica lo state_dict dal checkpoint
    checkpoint = torch.load(args.model_path)
    new_state_dict = {key.replace('model.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load(args.model_path))
    model.to("cuda")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    model.eval()

    # dummy_label = torch.randint(0, args.num_classes, (args.batch_size, 1, args.image_size, args.image_size * 2))
    # dummy_cond = {
    #     'label': dummy_label,
    #     'label_ori': dummy_label.float() * 255.0  # Fittizio
    # }
    # model_kwargs = preprocess_input_FDS(args, dummy_cond, num_classes=args.num_classes)
    random_y = torch.randint(0, args.num_classes + 1, (args.batch_size, args.num_classes + 1, args.image_size, args.image_size * 2)).float().to("cuda")
    model_kwargs = {
        'y': random_y,  # Mappa semantica randomica
        's': args.s     # Parametro iper
    }

    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop

    # Profilazione con il profiler nativo di PyTorch
    print("Profiling model with PyTorch Profiler...")

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA],  # Profilazione CPU e GPU
        record_shapes=True,  # Registra le forme dei tensori
        profile_memory=True  # Misura anche l'utilizzo della memoria
    ) as prof:
        with profiler.record_function("model_inference"):
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size * 2),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                prossgress=False
            )

    # Stampa i risultati del profiling
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # Salva i risultati in un file di traccia per una visualizzazione più dettagliata (ad esempio, con TensorBoard)
    prof.export_chrome_trace("trace.json")
    

def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        num_classes=35,
        s=1.0,
        snr=100,
        pool="med",
        add_noise=False,
        noise_to="semantics",
        unet_model="unet" #"unet", "spadeboth", "enconly"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()