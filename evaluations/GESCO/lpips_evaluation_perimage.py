import os
import torch
import pathlib
import argparse
from PIL import Image
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='Cuda or CPU')
parser.add_argument('--path1', type=str, default='./RESULTS', help='Original images path')
parser.add_argument('--path2', type=str, default='./RESULTS', help='Generated images path')
opt = parser.parse_args()

lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, compute_on_cpu=False)
lpips_metric = lpips_metric.to(opt.device)

path1 = pathlib.Path(opt.path1)
data = list(path1.glob('*.jpg')) + list(path1.glob('*.png'))
path2 = pathlib.Path(opt.path2)
recon = list(path2.glob('*.jpg')) + list(path2.glob('*.png'))

for idx in range(min(len(recon), len(data))):
    img1 = torchvision.io.read_image(str(data[idx]))
    img2 = torchvision.io.read_image(str(recon[idx]))
    # img1 = (img1 - torch.min(img1) / (torch.max(img1) - torch.min(img1)))
    # img2 = (img2 - torch.min(img2) / (torch.max(img2) - torch.min(img2)))
    img1 = img1/255
    img2 = img2/255

    eval_lpips = lpips_metric(img1.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())

print(eval_lpips)