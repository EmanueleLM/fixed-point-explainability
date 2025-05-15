# ------------------------------------------------------------------------------
# This code is adapted from:
# https://github.com/SrishtiGautam/ProtoVAE
# Original license: MIT
# 
# NOTE:
# If launched as a notebook, modify the arguments in settings.py

# %%
import sys
import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from helpers import makedir
import model
import train_and_test as tnt
import save
import matplotlib.pyplot as plt
import numpy as np
import dataloader_qd as dl
from settings import *
from matplotlib.pyplot import show

if is_local_env(): root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
else: root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from RecexpPrototype import Recexp

# %%
# usage example (from cmd)
# python3 protovae/eval_protovae.py -data mnist -idx 0
if is_local_env(): 
    os.chdir("../")
print("Working in", os.getcwd())

# %%
torch.manual_seed(12345+idx)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

# %%
model_dir = f"results/protovae/{data_name}_npr_{num_prototypes}_idx_{idx}/model.pth" 
model_file = model_dir
makedir(model_dir)
prototype_dir = model_dir + 'prototypes' + "/"
makedir(prototype_dir)


# %%
if (data_name == "mnist"):
    mean = (0.5)
    std = (0.5)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.MNIST(root=data_path, train=True,
                              download=True, transform=transform)
    testset = datasets.MNIST(root=data_path, train=False,
                             download=True, transform=transform)

elif (data_name == "fmnist"):
    mean = (0.5)
    std = (0.5)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.FashionMNIST(root=data_path, train=True,
                              download=True, transform=transform)
    testset = datasets.FashionMNIST(root=data_path, train=False,
                             download=True, transform=transform)


# %%
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

test_loader_expl = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=num_workers)

# %%
print('data : ',data_name)
print('training set size: {0}'.format(len(train_loader.dataset)))
print('test set size: {0}'.format(len(test_loader.dataset)))

# %%
jet = False
if(data_name == "mnist" or data_name=="fmnist"):
    jet = True

# %%
# Load the model
protovae = model.ProtoVAE().to(device)
protovae.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)

# %%
#Testing
protovae.eval()
test_acc, test_ce, test_recon, test_kl, test_ortho = tnt.test(model=protovae, dataloader=test_loader)


# %%
## Retrieve the learned prototypes
prototype_imgs = protovae.get_prototype_images()
n_prototypes = len(prototype_imgs)
num_p_per_class = protovae.num_prototypes_per_class


# %%
# Imlement the required abstract functions for Recexp (see RecexpPrototype.py for details)
class ProtovaeRecexp(Recexp):
    def get_input_and_label(self, idx):
        orig_input, orig_label_int = test_loader_expl.dataset[idx]
        return orig_input, orig_label_int

    def get_prediction_and_distance(self, input_data):
        ndim = input_data.dim()
        if ndim==3:
            input_data = input_data.clone().unsqueeze(0)
        with torch.no_grad():
            logits, proto_sim = protovae.pred_class(input_data.to(device))
            proto_dist = protovae.distance_2_similarity(proto_sim)

        if ndim==3:
            return logits[0].cpu().numpy(), proto_dist[0].cpu().numpy()
        else:
            return logits.cpu().numpy(), proto_dist.cpu().numpy()

    def get_pred_class(self, logits):
        return int(np.argmax(logits))

    def copy_input(self, input_tensor):
        return input_tensor.detach().clone()

    def get_prototypes(self):
        return prototype_imgs

    def get_prototype_at_idx(self, idx):
        return prototype_imgs[idx]

    def show_input(self, input_data):
        plt.imshow(input_data.detach().cpu().reshape(28,28), cmap="gray")
        plt.axis("off")
        plt.gcf().set_size_inches(1,1)
        plt.show()

    def plot_prototypes(self, order_idx, distances=None):
        nrow, ncol = 1, len(order_idx)
        figsize = (19,5) if len(order_idx)==self.n_prototypes else (1,4)
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
        for i in range(len(order_idx)):
            ax = axs[i] if nrow==1 else axs[i//ncol,i%ncol]
            ax.imshow(prototype_imgs[order_idx[i],:].clone().detach().cpu().numpy().reshape(28,28), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if not (distances is None):
                title_text = distances[order_idx[i]].round(decimals=1) 
                axs[i].set_title(title_text, y=0.95)
        plt.show()



# %%
# Instantiate the recursive explainer
recexp = ProtovaeRecexp(n_prototypes)

# %%
recexp.plot_prototypes(np.arange(num_prototypes))

# %%
# Compute class preservation (ALLOWING self-references)
_ = recexp.check_prototype_class_preservation(discard_previous=False, verbose=False)

# %%
# Compute class preservation (PREVENTING self-references)
_ = recexp.check_prototype_class_preservation(discard_previous=True, verbose=False)

# %%
# Check self-consistency
next_list_nodiscard = recexp.get_next_list()
self_cons_nodiscard_perc = np.sum(next_list_nodiscard==np.arange(n_prototypes))/n_prototypes*100
print(f"Self-consistency: {self_cons_nodiscard_perc}\%")

savepath = f"selfcons_protovae_{data_name}_idx_{idx}.pdf"
recexp.plot_self_consistency(next_list_nodiscard, savepath=savepath, show=True)

# %%
# Evaluate recursive explanations for the test set (ALLOWING self-references)
num_inputs = test_loader_expl.dataset.data.shape[0] #process the entire test set
max_steps  = n_prototypes+10  #maximum number of iterations (recursions) for each input
verbose= is_local_env() and False

recexp.evaluate(num_inputs, max_steps, allow_self_references=True, verbose=verbose)

# %%
print("Stats [allow_self_references=True]")
recexp.print_stats()


# %%
# Evaluate recursive explanations for the test set (PREVENTING self-references)
recexp.evaluate(num_inputs, max_steps, allow_self_references=False, verbose=verbose)

# %%
print("Stats [allow_self_references=False]")
recexp.print_stats()

# %%

# %%

# %%


if is_local_env():
    # Final step -- Visualize inputs for which class preservation is broken
    # Functions re-adapted for convenience
    def show_input(image, title=None, savepath=None):
        plt.imshow(image.reshape(28,28), cmap="gray")
        plt.axis("off")
        plt.gcf().set_size_inches(1,1)
        plt.tight_layout()
        if title is not None: plt.title(title)
        if savepath is not None: plt.savefig(savepath, dpi=300)
        plt.show()

    def plot_prototypes(order_idx, distances=None, savepath=None):
        nrow, ncol = 1, len(order_idx)
        figsize = (19,5) if len(order_idx)==n_prototypes else (10,3)
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
        for i in range(len(order_idx)):
            ax = axs[i] if nrow==1 else axs[i//ncol,i%ncol]
            ax.imshow(prototype_imgs[order_idx[i],:].clone().detach().cpu().numpy().reshape(28,28), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if not (distances is None):
                title_text = distances[order_idx[i]].detach().cpu().numpy()
                title_text = title_text.round(2)
                axs[i].set_title(title_text, y=0.95)
        if savepath is not None: plt.savefig(savepath, dpi=300)
        plt.show()

    show_max=3
    num_protos_plot=10 #visualize the first prototypes
    #num_protos_plot=num_prototypes # use to visualize all the prototypes
    if is_local_env():
        for i, idx in enumerate( np.where(recexp.class_preserv_broken_array)[0] ):
            if i==show_max: break
            print(f"idx: {idx}, class: {test_loader_expl.dataset.targets[idx]}")
            cur_input_tmp = test_loader_expl.dataset[idx][0].unsqueeze(0)
            pred_tmp, sim_tmp = protovae.pred_class(cur_input_tmp.to(device))
            dist_tmp = protovae.distance_2_similarity(sim_tmp)[0]
            print("Pred", pred_tmp.argmax())
            proto_order_tmp = dist_tmp.argsort().cpu().numpy() 
            show_input(cur_input_tmp.detach().cpu(), title="Input", savepath=f"input_{idx}.pdf")
            print(dist_tmp)
            plot_prototypes(proto_order_tmp[0:num_protos_plot], dist_tmp, savepath=f"expl_{idx}.pdf")

