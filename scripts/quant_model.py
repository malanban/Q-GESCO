"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.distributed as dist
import torchvision as tv
from scipy import ndimage, signal
from pooling import MedianPool2d

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data

# from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from QDrop.quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
    set_act_quantize_params,
)

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

def random_calib_data_generator(shape, num_samples, device,class_cond=True):
    logger.log('Calibration Data Generation...')
    calib_data = []
    for batch in range(num_samples):
        img = torch.randn(*shape, device=device)
        calib_data.append(img)
    t = torch.tensor([1] * num_samples, device=device)  # TODO timestep gen
    if class_cond:
        cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
        return torch.cat(calib_data, dim=0), t, cls
    else:
        return torch.cat(calib_data, dim=0), t


def quant_model(args, cnn):
    # build quantization parameters
    wq_params = {
        "n_bits": args.n_bits_w,
        "channel_wise": args.channel_wise,
        "scale_method": args.init_wmode,
        "symmetric": True,
    }
    aq_params = {
        "n_bits": args.n_bits_a,
        "channel_wise": False,
        "scale_method": args.init_amode,
        "leaf_param": True,
        "prob": args.prob,
        "symmetric": True,
    }

    qnn = QuantModel(
        model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params
    )
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print("Setting the first and the last layer to 8-bit")
        qnn.set_first_last_layer_to_8bit()

    qnn.disable_network_output_quantization()
    print("check the model!")
    print(qnn)
    cali_data = random_calib_data_generator(
        [1, 3, args.image_size, args.image_size], args.calib_num_samples, "cuda", args.class_cond
    )
    device = next(qnn.parameters()).device
    # print('the quantized model is below!')
    # Kwargs for weight rounding calibration
    assert args.wwq is True
    kwargs = dict(
        cali_data=cali_data,
        iters=args.iters_w,
        weight=args.weight,
        b_range=(args.b_start, args.b_end),
        warmup=args.warmup,
        opt_mode="mse",
        wwq=args.wwq,
        waq=args.waq,
        order=args.order,
        act_quant=args.act_quant,
        lr=args.lr,
        input_prob=args.input_prob,
        keep_gpu=not args.keep_cpu,
    )

    if args.act_quant and args.order == "before" and args.awq is False:
        """Case 2"""
        set_act_quantize_params(
            qnn, cali_data=cali_data, awq=args.awq, order=args.order
        )

    """init weight quantizer"""
    set_weight_quantize_params(qnn)
    if not args.use_adaround:
        set_act_quantize_params(
            qnn, cali_data=cali_data, awq=args.awq, order=args.order
        )
        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn
    else:

        def set_weight_act_quantize_params(module):
            if isinstance(module, QuantModule):
                layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                block_reconstruction(qnn, module, **kwargs)
            else:
                raise NotImplementedError

        def recon_model(model: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    print("Reconstruction for layer {}".format(name))
                    set_weight_act_quantize_params(module)
                elif isinstance(module, BaseQuantBlock):
                    print("Reconstruction for block {}".format(name))
                    set_weight_act_quantize_params(module)
                else:
                    recon_model(module)

        # Start calibration
        logger.log('Starting Calibration..')
        recon_model(qnn)

        if args.act_quant and args.order == "after" and args.waq is False:
            """Case 1"""
            set_act_quantize_params(
                qnn, cali_data=cali_data, awq=args.awq, order=args.order
            )

        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn



def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    logger.configure()
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to("cuda")
    # if args.use_fp16:
    #     model.convert_to_fp16()
    model.eval()

    # quantize model
    model = quant_model(args, model)
    


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

    parser.add_argument(
        "--seed", default=1005, type=int, help="random seed for results reproduction"
    )

    # quantization parameters
    parser.add_argument(
        "--n_bits_w", default=4, type=int, help="bitwidth for weight quantization"
    )
    parser.add_argument(
        "--channel_wise",
        action="store_true",
        help="apply channel_wise quantization for weights",
    )
    parser.add_argument(
        "--n_bits_a", default=4, type=int, help="bitwidth for activation quantization"
    )
    parser.add_argument(
        "--act_quant", action="store_true", help="apply activation quantization"
    )
    parser.add_argument("--disable_8bit_head_stem", action="store_true")

    # weight calibration parameters
    parser.add_argument(
        "--calib_num_samples",
        default=1024,
        type=int,
        help="size of the calibration dataset",
    )
    parser.add_argument(
        "--iters_w", default=20000, type=int, help="number of iteration for adaround"
    )
    parser.add_argument(
        "--weight",
        default=0.01,
        type=float,
        help="weight of rounding cost vs the reconstruction loss.",
    )
    parser.add_argument(
        "--keep_cpu", action="store_true", help="keep the calibration data on cpu"
    )

    parser.add_argument(
        "--wwq",
        action="store_true",
        help="weight_quant for input in weight reconstruction",
    )
    parser.add_argument(
        "--waq",
        action="store_true",
        help="act_quant for input in weight reconstruction",
    )

    parser.add_argument(
        "--b_start",
        default=20,
        type=int,
        help="temperature at the beginning of calibration",
    )
    parser.add_argument(
        "--b_end", default=2, type=int, help="temperature at the end of calibration"
    )
    parser.add_argument(
        "--warmup",
        default=0.2,
        type=float,
        help="in the warmup period no regularization is applied",
    )

    # activation calibration parameters
    parser.add_argument("--lr", default=4e-5, type=float, help="learning rate for LSQ")

    parser.add_argument(
        "--awq",
        action="store_true",
        help="weight_quant for input in activation reconstruction",
    )
    parser.add_argument(
        "--aaq",
        action="store_true",
        help="act_quant for input in activation reconstruction",
    )

    parser.add_argument(
        "--init_wmode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for weight",
    )
    parser.add_argument(
        "--init_amode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for activation",
    )
    # order parameters
    parser.add_argument(
        "--order",
        default="before",
        type=str,
        choices=["before", "after", "together"],
        help="order about activation compare to weight",
    )
    parser.add_argument("--prob", default=1.0, type=float)
    parser.add_argument("--input_prob", default=1.0, type=float)
    parser.add_argument("--use_adaround", action="store_true")

    return parser


if __name__ == "__main__":
    main()