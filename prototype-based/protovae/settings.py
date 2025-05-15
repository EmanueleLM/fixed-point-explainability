# ------------------------------------------------------------------------------
# This code is adapted from:
# https://github.com/SrishtiGautam/ProtoVAE
# Original license: MIT

import argparse
import os
import sys

def is_local_env():
    """Returns True if running in an interactive environment"""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except:
        return False

if is_local_env():
    # If eval_prototype.py is executed as a notebook, adapt this configuration
    args = argparse.Namespace(
        data=['mnist'],  # Options: mnist, fmnist
        npr=[50],        # Number of prototyps
        mode=['train'],  # Not used to evaluate recursive explanations
        save_dir='/tmp/',
        model_file=['/tmp/protovae/model.pth'], # Use for train only (overwritten in eval_protovae.py)
        expl=[False],
        idx=[0] # Index of the experiment repetition
    )
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', nargs=1, type=str, default=['mnist'])
    parser.add_argument('-npr',  nargs=1, type=int, default=[50])
    parser.add_argument('-mode', nargs=1, type=str, default=['train'])
    parser.add_argument('-save_dir', type=str, default="/tmp/protovae/")
    parser.add_argument('-model_file', nargs=1, type=str, default=['/tmp/protovae/model.pth']) #test
    parser.add_argument('-expl', nargs=1, type=bool, default=[False])
    parser.add_argument('-idx',  nargs=1, type=int, default=[0])
    args = parser.parse_args()

# Retrieve the arguments for convenience
data_name = args.data[0]
num_prototypes = args.npr[0]
mode = args.mode[0]
save_dir = args.save_dir
model_file = args.model_file[0]
expl = args.expl[0]
idx = args.idx[0]


data_path = 'Data/'

coefs = {
        'crs_ent': 1,
        'recon': 1,
        'kl': 1,
        'ortho': 1,
    }


if (data_name == "mnist"):
    img_size = 28
    latent = 256
    #num_prototypes = 50
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10


elif (data_name == "fmnist"):
    img_size = 28
    latent = 256
    #num_prototypes = 100
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10
