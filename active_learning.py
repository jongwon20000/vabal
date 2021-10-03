'''Train CIFAR10 with PyTorch.'''
import random
import numpy as np
import time

import argparse

from dataset_load import datapool_sampling
from vabal_model import classification_modeler


# argument parser
parser = argparse.ArgumentParser(description='VaBAL Training')

parser.add_argument('--dataset_name', default='CIFAR10', type=str, help='name of training dataset')
parser.add_argument('--dataset_sampling_ratio', default=0.1, type=float, help='dataset sampling ratio')
parser.add_argument('--dataset_sampling_classes', default=[1,2,3,4,5,6,7,8,9], type=int, nargs="+", help='epoch when the learning rate changes')

parser.add_argument('--model_name', default='RESNET18', type=str, help='name of training model')

parser.add_argument('--batch_size', default=128, type=int, help='the size of mini-batch')
parser.add_argument('--init_lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--last_lr', default=0.01, type=float, help='last learning rate')
parser.add_argument('--epoch', default=200, type=int, help='entire epoch number')
parser.add_argument('--lr_change_epoch', default=160, type=int, help='epoch when the learning rate changes')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum value for momentum SGD')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay factor for loss')

parser.add_argument('--ckpt_folder_name', default='', type=str, help='folder name for checkpoint saving')
parser.add_argument('--rand_seed', default=0, type=int, help='random seed value')

# active learning arguments
parser.add_argument('--sample_size', default=500, type=int, help='sampling size for every round')
parser.add_argument('--rounds', default=6, type=int, help='maximum sampling rounds')

# VAE arguments
parser.add_argument('--encode_size', default=128, type=int, help='vae encoding size')
parser.add_argument('--fc_size', default=128, type=int, help='vae fc size')
parser.add_argument('--class_latent_size', default=10, type=int, help='vae latent size per class')
parser.add_argument('--vae_init_lr', default=0.01, type=float, help='vae initial learning rate')
parser.add_argument('--vae_last_lr', default=0.001, type=float, help='vae last learning rate')
parser.add_argument('--vae_lr_change_epoch', default=16, type=int, help='vae epoch when the learning rate changes')
parser.add_argument('--vae_epoch', default=20, type=int, help='vae entire epoch number')
parser.add_argument('--vae_lambda', default=0.005, type=float, help='vae loss lambda factor')
parser.add_argument('--sampling_num', default=100, type=int, help='number of sampling for probabilistic inference')

parser.add_argument('--random', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

if not args.ckpt_folder_name:
    args.ckpt_folder_name = '{:}'.format(time.time())

# random seed fix
random.seed(args.rand_seed)

print(args)

# initial sample pick
(datapool_idx, trainset_len, testset_len) = datapool_sampling(args.dataset_name, args.dataset_sampling_classes, args.dataset_sampling_ratio)
sample_idx_all = list(range(trainset_len))
random.shuffle(sample_idx_all)
sample_idx = sample_idx_all[:args.sample_size] # index list for sampled samples
val_idx = [x for x in range(trainset_len) if x not in sample_idx] # index list for the remaining samples (unlabelled)

for i_rounds in range(args.rounds):
    
    print('{}-th round : {} samples! starts...'.format(i_rounds, len(sample_idx)) )
    
    # trainer build-up
    classification_module = classification_modeler(args, datapool_idx, sample_idx)

    # trainer training
    classification_module.train(args.random)
    
    # trainer saving
    folder_name = classification_module.save(args.ckpt_folder_name + '_{:02d}'.format(i_rounds))
    np.save(folder_name + '/sample_idx.npy', sample_idx)

    # trainer testing
    classification_module.test()
    
    # validation set scoring & logging
    if not args.random:
        scores = classification_module.val()
        np.save(folder_name + '/scores.npy', scores)        
    else:
        scores = [random.random() for _ in range(len(val_idx))]
    
    # active sampling (Update sample_idx & val_idx)
    sample_idx = sample_idx + [val_idx[idx] for idx in np.argsort(scores)[-args.sample_size:]]
    val_idx = [x for x in range(trainset_len) if x not in sample_idx]

