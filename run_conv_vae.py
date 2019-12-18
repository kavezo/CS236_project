# Copyright (c) 2018 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.conv_vae import ConvVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=128,    help="Number of latent dimensions") # Japanese has 20 phonemes, times three positions
parser.add_argument('--iter_max',  type=int, default=5000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=500, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=2,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=0,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'convvae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_hiyori_data(device, use_test_subset=True)
vae = ConvVAE(z_dim=args.z, name=model_name, k=6).to(device)

if args.train:
    #ut.load_model_by_name(vae, global_step=args.iter_max, device=device)
    writer = ut.prepare_writer(model_name, overwrite_existing=False)
    train(model=vae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    ut.evaluate_lower_bound_continuous(vae, labeled_subset, run_iwae=args.train == 2)

else:
    ut.load_model_by_name(vae, global_step=args.iter_max, device=device)
    ut.evaluate_lower_bound_continuous(vae, labeled_subset, run_iwae=True)
