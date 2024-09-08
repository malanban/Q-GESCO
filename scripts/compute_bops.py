"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import csv

import torch
import torch.profiler

# import torch.distributed as dist
# import torch.autograd.profiler as profiler
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

    # with profiler.profile(record_shapes=True, with_flops=True, use_device='cuda') as prof:  # Usa solo record_shapes per tracciare le forme
    #     with profiler.record_function("model_inference"):
    #         sample = sample_fn(
    #             model,
    #             (args.batch_size, 3, args.image_size, args.image_size * 2),
    #             clip_denoised=args.clip_denoised,
    #             model_kwargs=model_kwargs,
    #             progress=False
    #         )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # record_shapes=True,
        # profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as profiler:
        sample_fn (
            model,
            (args.batch_size, 3, args.image_size, args.image_size * 2),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=False
        )                

    # Raccogliere i risultati del profiler
    events = profiler.key_averages()

    # Ordinare per FLOPs
    valid_flops_events = [event for event in events if event.flops > 0]
    sorted_events = sorted(valid_flops_events, key=lambda e: e.flops, reverse=True)

    # Calcolare il numero totale di FLOPs
    total_flops = sum([event.flops for event in sorted_events])

    # Stampare il numero totale di FLOPs
    print(f"Numero totale di FLOPs: {total_flops}\n")
    # print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # # Stampa i risultati del profiling
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    # events = prof.events()
    # flops = sum([int(evt.flops) for evt in events]) 
    # print("flops: ", flops)
    # # Salva i risultati in un file di traccia per una visualizzazione più dettagliata (ad esempio, con TensorBoard)
    # prof.export_chrome_trace("trace.json")
    # Raccogliere i risultati del profiler
    # events = profiler.key_averages()

    # Ordinare per FLOPs e filtrare solo quelli che hanno GFLOPs definiti

    # Stampare i risultati filtrati
    print(f"{'Operation':<30} {'FLOPs':<20} {'Self CUDA Time (ns)':<20} {'CPU Time (ns)':<20}")
    for event in sorted_events:
        print(event)
        # print(f"{event.key:<30} {event.flops:<20} {event.self_cuda_time_total:<20} {event.self_cpu_time_total:<20}")

    # Salvataggio dei dati filtrati in un file CSV
    csv_file = 'flops_profile_filtered.csv'

    # Definire le intestazioni per il file CSV
    fieldnames = ['Operation', 'GFLOPs', 'Self CUDA Time (ns)', 'CPU Time (ns)']

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for event in valid_flops_events:
            writer.writerow({
                'Operation': event.key,
                'GFLOPs': event.flops,
                'Self CUDA Time (ns)': event.self_cuda_time_total,
                'CPU Time (ns)': event.self_cpu_time_total
            })

    print(f"I dati filtrati dei FLOPs sono stati salvati nel file {csv_file}.")
    

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