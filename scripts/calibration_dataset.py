# @pineatus
import torch
from guided_diffusion.image_datasets import load_data
from .quant_model import preprocess_input_FDS

def generate_t(args, t_mode, num_samples, diffusion, device):
    if t_mode == "1":
        t = torch.tensor([1] * num_samples, device=device)
    elif t_mode == "-1":
        t = torch.tensor(
            [diffusion.num_timesteps - 1] * num_samples, device=device
        )  
    elif t_mode == "mean":
        t = torch.tensor(
            [diffusion.num_timesteps // 2] * num_samples, device=device
        )  
    elif t_mode == "manual":
        t = torch.tensor(
            [diffusion.num_timesteps * 0.1] * num_samples, device=device
        )  
    elif t_mode == "normal":
        shape = torch.Tensor(num_samples)
        normal_val = torch.nn.init.normal_(shape, mean=args.calib_t_mode_normal_mean, std=args.calib_t_mode_normal_std)*diffusion.num_timesteps
        # TODO: @pineatus 
        # normal_val = torch.nn.init.normal_(shape, mean=args.calib_t_mode_normal_mean*diffusion.num_timesteps, std=math.sqrt(args.calib_t_mode_normal_std*diffusion.num_timesteps))
        t = normal_val.clone().type(torch.int).to(device=device)
        # print(t.shape)
        # print(t[0:30])
    elif t_mode == "random":
        # t = torch.randint(0, diffusion.num_timesteps, [num_samples], device=device)
        t = torch.randint(0, int(diffusion.num_timesteps*0.8), [num_samples], device=device)
        print(t.shape)
        print(t)
    elif t_mode == "uniform":
        t = torch.linspace(
            0, diffusion.num_timesteps, num_samples, device=device
        ).round()
    else:
        raise NotImplementedError
    return t.clamp(0, diffusion.num_timesteps - 1)

# @pineatus
def random_calib_data_generator(shape, num_samples, diffusion, device, data, args, t_mode):
    """
    Questa funzione genera il calibration dataset. 
    Calibration Datas sono costituiti da:
        - una noisy image
        - input_semantic generato del blocco FDS
        - timestep t
    """
    # add num_sample as first dimension for the calibration_data tensor
    new_shape = (num_samples, *shape)
    # Generate random noisy images calibration data
    calib_data = torch.randn(*new_shape, device=device)
    # Generate Timestesps
    t = generate_t(t_mode, num_samples, diffusion, device)
    # Generate Input_Semantics
    input_semantics = []
    for i, (batch, cond) in enumerate(data):
        model_kwargs = preprocess_input_FDS(args, cond, num_classes=args.num_classes, one_hot_label=args.one_hot_label)
        input_semantics.append(model_kwargs['y'])
        if ((i+1) * args.batch_size >= num_samples):
            break
    print(calib_data.shape, input_semantics[0].shape)
    return calib_data, t, torch.cat(input_semantics[:num_samples], dim=0)


# def random_calib_data_generator(
#     shape, num_samples, device, t_mode, diffusion, class_cond=True
# ):
#     calib_data = []
#     for batch in range(num_samples):
#         img = torch.randn(*shape, device=device)
#         calib_data.append(img)
#     t = generate_t(t_mode, num_samples, diffusion, device)
#     t = diffusion._scale_timesteps(t)
#     if class_cond:
#         cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
#         return torch.cat(calib_data, dim=0), t, cls
#     else:
#         return torch.cat(calib_data, dim=0), t


def raw_calib_data_generator(
    args, num_samples, device, t_mode, diffusion, class_cond=True
):
    loader = load_data(
        data_dir=args.data_dir,
        batch_size=num_samples,
        image_size=args.image_size,
        class_cond=class_cond,
    )
    calib_data, cls = next(loader)
    calib_data = calib_data.to(device)
    t = generate_t(t_mode, num_samples, diffusion, device)
    t = diffusion._scale_timesteps(t)
    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t


def forward_t_calib_data_generator(
    args, num_samples, device, t_mode, diffusion, class_cond=True
):
    loader = load_data(
        data_dir=args.data_dir,
        batch_size=num_samples,
        image_size=args.image_size,
        class_cond=class_cond,
    )
    calib_data, cls = next(loader)
    calib_data = calib_data.to(device)
    t = generate_t(t_mode, num_samples, diffusion, device).long()
    x_t = diffusion.q_sample(calib_data, t)
    t = diffusion._scale_timesteps(t)
    if class_cond:
        return x_t, t, cls.to(device)
    else:
        return x_t, t


# def backward_t_calib_data_generator(
#     model, args, num_samples, device, t_mode, diffusion, class_cond=True
# ):
#     model_kwargs = {}
#     if class_cond:
#         cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
#         model_kwargs["y"] = cls
#     loop_fn = (
#         diffusion.ddim_sample_loop_progressive
#         if args.use_ddim
#         else diffusion.p_sample_loop_progressive
#     )
#     t = generate_t(args, t_mode, num_samples, diffusion, device).long()
#     calib_data = None
#     for now_rt, sample_t in enumerate(
#         loop_fn(
#             model,
#             (num_samples, 3, args.image_size, args.image_size),
#             clip_denoised=args.clip_denoised,
#             model_kwargs=model_kwargs,
#             device=device,
#         )
#     ):
#         sample_t = sample_t["sample"]
#         if calib_data is None:
#             calib_data = torch.zeros_like(sample_t)
#         mask = t == now_rt
#         if mask.any():
#             calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
#     calib_data = calib_data.to(device)
#     t = diffusion._scale_timesteps(t)
#     if class_cond:
#         return calib_data, t, cls.to(device)
#     else:
#         return calib_data, t

# @pineatus
def backward_t_calib_data_generator(
    model, args, num_samples, device, t_mode, diffusion, loader, class_cond=True
):
    # Generate timestep t
    t = generate_t(args, t_mode, num_samples, diffusion, device).long()

    # Generate conditional signal y
    model_kwargs = {}
    if class_cond:
        input_semantics = []
        for i, (_, cond) in enumerate(loader):
            model_kwargs = preprocess_input_FDS(args, cond, num_classes=args.num_classes, one_hot_label=args.one_hot_label)
            input_semantics.append(model_kwargs['y'])
            if ((i+1) * args.batch_size >= num_samples):
                break
        model_kwargs['y'] = torch.cat(input_semantics[:num_samples], dim=0)

    # Generate sample 
    """
    TODO: the loop_fn is derived from PTQ4DM/improved-diffusion. Check compatibility with guided-diffusion
    """
    loop_fn = (
        diffusion.ddim_sample_loop_progressive
        if args.use_ddim
        else diffusion.p_sample_loop_progressive
    )
    calib_data = None
    for now_rt, sample_t in enumerate(
        loop_fn(
            model,
            (num_samples, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
        )
    ):
        sample_t = sample_t["sample"]
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
    calib_data = calib_data.to(device)
    t = diffusion._scale_timesteps(t)
    if class_cond:
        return calib_data, t, model_kwargs['y'].to(device)
    else:
        return calib_data, t
