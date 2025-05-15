import os
import itertools

if __name__ == '__main__':

    # Configurations
    dataset_vec = ["mnist"] #["mnist", "fmnist"]
    npr_vec     = [50]      #[10, 20, 50, 100] 
    exp_idx_vec = [0]       #[0]

    save_folder = "results/protovae/"
    script_name = "launch/train_protovae.sh"

    for (dataset, npr, exp_idx) in itertools.product(dataset_vec, npr_vec, exp_idx_vec):
    
        # Results folder
        cur_save_folder = save_folder + "%s_npr_%d_idx_%d/"%(dataset, npr, exp_idx)

        # LAUNCH
        cmd = "sh " + script_name
        cmd = cmd + " " + dataset           #-data
        cmd = cmd + " " + str(npr)          #-npr
        cmd = cmd + " " + str(exp_idx)      #-idx
        cmd = cmd + " " + cur_save_folder   #-save_dir

        # Launch process
        print(cmd)
        os.system(cmd)




