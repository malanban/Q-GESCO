"""
Generate the Calibration Dataset for the quantization process.

The Dataset is generated according to the algorithm described in Q-Diffusion at: https://arxiv.org/abs/2302.04304

"""
import argparse
import os

import torch as th
# import torch.distributed as dist
# import torchvision as tv

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
from pooling import MedianPool2d

from pytorch_lightning import seed_everything

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# SNR (var): 1 (0.9) 5 (0.6) 10 (0.36) 15 (0.22) 20 (0.13) 25 (0.08) 30 (0.05) 100 (0.0)
# SNR_DICT = {
#     100: 0.0,
#     30: 0.05,
#     25: 0.08,
#     20: 0.13,
#     15: 0.22,
#     10: 0.36,
#     5: 0.6,
#     1: 0.9
# }
SNR_DICT = {
    100: 0.0,
    20: 0.13,
    10: 0.36,
    1: 0.9
}

def preprocess_input_FDS(data, num_classes, snr, one_hot_label=True):
    pool = "max"
    label_map = data['label'].long()

    # create one-hot label map
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    print("label map shape:", label_map.shape)

    input_semantics = input_label.scatter_(1, label_map, 1.0)
    map_to_be_discarded = []
    map_to_be_preserved = []
    input_semantics = input_semantics.squeeze(0)
    for idx, segmap in enumerate(input_semantics.squeeze(0)):
        if 1 in segmap:
            map_to_be_preserved.append(idx)
        else:
            map_to_be_discarded.append(idx)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        print("instance edge map shape: ", instance_edge_map.shape)
        input_semantics = th.cat((input_semantics.unsqueeze(0), instance_edge_map), dim=1)
        map_to_be_preserved.append(num_classes)
        num_classes += 1

    input_semantics = input_semantics[0][map_to_be_preserved]

    # Add noise based on the provided SNR
    noise = th.randn(input_semantics.shape, device=input_semantics.device) * snr
    input_semantics += noise

    # Apply pooling (median, mean, or max)
    if pool == "med":
        print("Using Median filter")
        med_filter = MedianPool2d(padding=1, same=True)
        input_semantics_clean = med_filter(input_semantics)
    elif pool == "mean":
        print("Using Average filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        input_semantics_clean = avg_filter(input_semantics)
    elif pool == "max":
        print("Using Max filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        max_filter = th.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        input_semantics_clean = max_filter(avg_filter(input_semantics))
    else:
        input_semantics_clean = input_semantics

    input_semantics_clean = input_semantics_clean.unsqueeze(0)

    input_semantics = th.empty(size=(input_semantics_clean.shape[0], num_classes, input_semantics_clean.shape[2], input_semantics_clean.shape[3]), device=input_semantics_clean.device)
    input_semantics[0][map_to_be_preserved] = input_semantics_clean[0]
    input_semantics[0][map_to_be_discarded] = th.zeros((len(map_to_be_discarded), input_semantics_clean.shape[2], input_semantics_clean.shape[3]), device=input_semantics_clean.device)

    print(f'input_semantic shape: {input_semantics.shape}')
    return {'y': input_semantics}

def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

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
        # unet_model="unet" #"unet", "spadeboth", "enconly"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    
    # Calibration specific Configs:
    # parser.add_argument(
    #     "--timesteps", type=int, default=1000, help="number of steps involved"
    # )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_snr", type=str, default="fixed", 
        choices=["fixed", "random", "uniform"], 
        help="SNR mode: fixed (default), random, or uniform"
    )
    parser.add_argument(
        "--cali_offset", type=int, default=0, 
        help="Offset for data_loader"
    )

    return parser

if __name__ == "__main__":
    # parse_args
    args = create_argparser().parse_args()

    # fix random seed
    seed_everything(args.seed)

    # Instanziate the Model
    print("Creating Model and Diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # Load the state_dict from checkpoint
    checkpoint = th.load(args.model_path)
    new_state_dict = {key.replace('model.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(th.load(args.model_path))
    if args.use_fp16:
        model.convert_to_fp16()
    model.to("cuda")
    model.eval()

    print("Creating Data Loader...")
    data_loader = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        random_crop=False,
        random_flip=False,
        is_train=False
    )

    # Sampling Procedure:
    print("Start Sampling")
    device = "cuda:0"
    T = args.diffusion_steps    # Total Timesteps
    N = args.cali_n             # Number of Samples for each Timestep
    ds = int(T / args.cali_st)  # Sampling Interval
    loop_fn = (
        diffusion.ddim_sample_loop_progressive
        if args.use_ddim
        else diffusion.p_sample_loop_progressive
    )

    # Get SNR Values
    snr_values = list(SNR_DICT.values())
    num_snr_values = len(snr_values)
    total_batches = N // args.batch_size
        
    # uniform snr: Compute how many batches to associate for each different snr value
    if args.cali_snr == "uniform":
        repeat_per_snr = total_batches // num_snr_values
        remaining_batches = total_batches % num_snr_values

    xs_l, ts_l, cs_l = [], [], []
    
    for batch_idx, (images, cond) in enumerate(data_loader):
        if (batch_idx * args.batch_size >= args.cali_n + args.cali_offset):
            break
        if (batch_idx >= args.cali_offset):
            print(f'Processing batch {batch_idx}...')    

            # Compute the snr value based on provided strategy:
            if args.cali_snr == "fixed":
                # fixed: use fixed args.snr
                snr_value = SNR_DICT[args.snr]

            elif args.cali_snr == "random":
                # random: use random choice of snr_values
                snr_value = np.random.choice(snr_values)

            elif args.cali_snr == "uniform":
                # uniform: distribute possible snr values among batches
                snr_index = batch_idx // repeat_per_snr
                if snr_index >= num_snr_values:
                    # for remaining batches, use random index
                    snr_index = np.random.randint(0, num_snr_values)
                snr_value = snr_values[snr_index]

            # generate model_kwargs
            model_kwargs = preprocess_input_FDS(cond, num_classes=args.num_classes, snr=snr_value, one_hot_label=args.one_hot_label)
            model_kwargs['s'] = args.s

            # Intermediate Samples Generation Loop:
            for t, sample_t in enumerate(
                loop_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size * 2),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=True
                )
            ):
                if (t + 1) % ds == 0:
                    print(f't = {t}')
                    xs_l.append(sample_t['sample'])
                    ts_l.append((th.ones(args.batch_size) * t).float() * (1000.0 / T))
                    cs_l.append(model_kwargs['y'])

    data = {
        'xs': th.cat(xs_l, 0),
        'ts': th.cat(ts_l, 0),
        'cs': th.cat(cs_l, 0)
    }

    print("Sampling Complete")
    print(f'xs shape: {data["xs"].shape}')
    print(f'ts shape: {data["ts"].shape}')
    print(f'cs shape: {data["cs"].shape}')

    # snr label for filename
    if args.cali_snr == "fixed":
        snr_label = f"snr{args.snr}"
    else:
        snr_label = f'snr-{args.cali_snr}'

    # dynamic filename
    filename = f"cali_data_T{args.diffusion_steps}_n{args.cali_n}_st{args.cali_st}_{snr_label}_of{args.cali_offset}.pth"
    cali_data_path = os.path.join(args.results_path, filename)

  # Save the Calibration Dataset
    th.save(data, cali_data_path)
    print(f"Calibration Dataset saved in '{cali_data_path}'")
