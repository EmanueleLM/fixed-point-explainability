import os
import os.path
import sys
import itertools
import tracker

if __name__ == '__main__':

    # Configuration
    dataset_vec      = ["mnist"] #["mnist", "fashion"]
    n_prototypes_vec = [50]      #[10, 20, 50, 100] 
    exp_idx_vec      = [0]       #[0]

    save_folder  = "results/prototypeDNN/"
    script_name  = "launch/train_prototypeDNN.sh"

    for (n_prototypes, exp_idx, dataset) in itertools.product(n_prototypes_vec, exp_idx_vec, dataset_vec):
    
        # Results folder
        cur_save_folder = save_folder + "%s_npr_%d_cae_%d/"%(dataset, n_prototypes, exp_idx)

        # LAUNCH
        ####################
        cmd = "sh " + script_name
        cmd = cmd + " " + dataset           #-ds
        cmd = cmd + " " + str(n_prototypes) #-npr
        cmd = cmd + " " + str(exp_idx)      #-idx
        cmd = cmd + " " + cur_save_folder   #-s

        # Launch process
        print(cmd)
        os.system(cmd)
