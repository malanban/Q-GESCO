"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse, os, datetime
import logging
import numpy as np
import tqdm

import torch as th
import torch
import torch.nn as nn
from torch.cuda import amp

import torch.distributed as dist
import torchvision as tv
import torchvision.utils as tvu

from pytorch_lightning import seed_everything

from guided_diffusion.image_datasets import load_data

# from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from pooling import MedianPool2d

from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples


logger = logging.getLogger(__name__)
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
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
        help="quantization mode to use"
    )
    parser.add_argument(
        "--max_images", type=int, default=50000, help="number of images to sample"
    )
    parser.add_argument("--use_pretrained", action="store_true")


    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument("--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )

    return parser

if __name__ == "__main__":    
    # parse_args
    args = create_argparser().parse_args()
    
    # fix random seed
    seed_everything(args.seed)

    # setup logger
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir)
    args.logdir = logdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    args.image_folder = imglogdir

    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    # set the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instanziate the GESCO Pretrained Model
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
        model.convert_to_fp16() #: potenziale conflitto
    model.to(device)
    model.eval()

    if args.ptq:
        # Quantize the model
        if args.quant_mode == 'qdiff':
                print('Starting Quantization Process')
                wq_params = {'n_bits': args.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
                aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': args.quant_act}
                if args.resume:
                    logger.info('Load with min-max quick initialization')
                    wq_params['scale_method'] = 'max'
                    aq_params['scale_method'] = 'max'
                if args.resume_w:
                    wq_params['scale_method'] = 'max'
                # Instantiate Quant Model (Wrapper)    
                qnn = QuantModel(
                    model=model, weight_quant_params=wq_params, act_quant_params=aq_params, 
                    sm_abit=args.sm_abit)
                qnn.to(device)
                qnn.eval()
                print('Quant Model Created')

                if args.resume:
                    image_size = args.image_size
                    channels = args.num_channels
                    # random calibration data
                    cali_data = (torch.randn(1, channels, image_size, image_size*2), torch.randint(0, 1000, (1,)))
                    #: Serve il segnale di condizionamento?
                else:
                    logger.info(f"Loading {args.cali_n} data for {args.cali_st} timesteps for calibration")
                    cali_data = torch.load(args.cali_data_path, map_location='cpu')
                    cali_xs = cali_data['xs']
                    cali_ts = cali_data['ts']
                    cali_cs = cali_data['cs'] if args.cond else None                        
                    logger.info(f"Calibration data shape: {cali_xs.shape} {cali_ts.shape} {cali_cs.shape if args.cond else None}")
                    if args.resume_w:
                        resume_cali_model(qnn, args.cali_ckpt, (cali_xs, cali_ts, cali_cs), False, cond=args.cond)
                    else:
                        logger.info("Initializing weight quantization parameters")
                        qnn.set_quant_state(True, False)
                        if args.cond:
                            _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda(), cali_cs[:1].cuda())
                        else:
                            _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda())
                        logger.info("Initializing has done!")

                    # Kwargs for weight rounding calibration
                    kwargs = dict(
                        cali_data=cali_data, batch_size=args.cali_batch_size, 
                        iters=args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                        warmup=0.2, act_quant=False, opt_mode='mse'
                    )

                    def recon_model(model):
                        """
                        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                        """
                        for name, module in model.named_children():
                            logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                            if isinstance(module, QuantModule):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of layer {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for layer {}'.format(name))
                                    layer_reconstruction(qnn, module, **kwargs)
                            elif isinstance(module, BaseQuantBlock):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of block {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for block {}'.format(name))
                                    block_reconstruction(qnn, module, **kwargs)
                            else:
                                recon_model(module)
                    if not args.resume_w:
                        logger.info("Doing weight calibration")
                        recon_model(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=False)
                    if args.quant_act:                 
                        logger.info("Doing activation calibration")   
                        # Initialize activation quantization parameters
                        qnn.set_quant_state(True, True)
                        with torch.no_grad():
                            inds = np.random.choice(cali_xs.shape[0], 64, replace=False)
                            if args.cond:
                                _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda(), cali_cs[inds].cuda())
                            else:    
                                _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda())
                        
                            if args.running_stat:
                                logger.info('Running stat for activation quantization')
                                qnn.set_running_stat(True)
                                for i in range(int(cali_xs.size(0) / 64)):
                                    if args.cond:
                                        _ = qnn(
                                                (
                                                cali_xs[i * 64:(i + 1) * 64].to(device), 
                                                cali_ts[i * 64:(i + 1) * 64].to(device),
                                                cali_cs[i * 64:(i + 1) * 64].to(device)
                                                )
                                            )
                                    else:
                                        _ = qnn(
                                            (cali_xs[i * 64:(i + 1) * 64].to(device), 
                                            cali_ts[i * 64:(i + 1) * 64].to(device)))
                                qnn.set_running_stat(False)
                        
                        kwargs = dict(
                            cali_data=cali_data, iters=args.cali_iters_a, act_quant=True, 
                            opt_mode='mse', lr=args.cali_lr, p=args.cali_p)   
                        recon_model(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=True)   
                    
                    # Saving:
                    logger.info("Saving calibrated quantized UNet model")
                    for m in qnn.model.modules():
                        if isinstance(m, AdaRoundQuantizer):
                            m.zero_point = nn.Parameter(m.zero_point)
                            m.delta = nn.Parameter(m.delta)
                        elif isinstance(m, UniformAffineQuantizer) and self.args.quant_act:
                            if m.zero_point is not None:
                                if not torch.is_tensor(m.zero_point):
                                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                                else:
                                    m.zero_point = nn.Parameter(m.zero_point)
                    torch.save(qnn.state_dict(), os.path.join(args.logdir, "ckpt.pth"))





                    


