import sys
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import copy

class Recexp(ABC):
    def __init__(self, n_prototypes):
        self.n_prototypes = n_prototypes

        self.num_inputs, \
        self.correctly_classified_array, \
        self.class_preserv_broken_array, \
        self.cycle_found_array, \
        self.cycle_period_array, \
        self.steps_converge_array = [None for _ in range(6)]

    @abstractmethod
    def get_input_and_label(self, idx):
        """ 
        Retrieve the input and corresponding label for the given index. 

        Parameters
        ----------
        idx : int
            Index of the example to retrieve.

        Returns
        -------
        input_tensor : torch.Tensor
            A tensor containing the input features, ready to be passed to the model.
            This should be preprocessed as required by the model (e.g., normalized, reshaped).

        label : int or torch.Tensor
            The ground truth label corresponding to the input.
        """
        pass

    @abstractmethod
    def get_prediction_and_distance(self, input_data):
        """
        Compute the prediction logits and distances to all prototypes for the given input.

        Parameters
        ----------
        input_data : array-like or torch.Tensor
            A batch of input examples. The shape and type should match the expected input
            of the model

        Returns
        -------
        logits : np.ndarray
            A NumPy array of shape (batch_size, num_classes) containing the prediction logits
            for each input example.

        distances : np.ndarray
            A NumPy array of shape (batch_size, num_prototypes) representing the distances
            between each input example and each learned prototype.
        """
        pass

    @abstractmethod
    def get_pred_class(self, logits):
        """ 
        Returns the predicted class index (integer) given prediction logits.
        """
        pass

    @abstractmethod
    def copy_input(self, input):
        """ Auxiliary function that returns a copy of the input"""
        pass

    @abstractmethod
    def get_prototypes(self):
        """ 
        Auxiliary function to get the set of (decoded) prototypes from the model
        """
        pass

    @abstractmethod
    def get_prototype_at_idx(self, proto_idx):
        """ 
        Auxiliary function to retrieve the prototype at the given index (proto_idx)
        """
        pass

    def show_input(self, input_data):
        """ Auxiliary function to plot an input sample. """
        pass  

    def plot_prototypes(self, order_idx, distances=None):
        """
        Plot the prototypes in a specified order, optionally displaying distances as titles.

        Parameters
        ----------
        order_idx : list or np.ndarray
            A list of prototype indices indicating the order in which the prototypes
            should be displayed.

        distances : optional
            An array of distances corresponding to each prototype. If provided,
            the title of each subplot will display the distance of the corresponding
            prototype (rounded to one decimal place). The shape should match the
            number of prototypes.
        """
        pass 



    def evaluate(self, num_inputs, max_steps, allow_self_references=True, verbose=False):
        """
        Implements the recursive-process for the number of inputs specified.

        Parameters
        ----------
        num_inputs : int
            Number of input examples to evaluate. This controls the size of the evaluation loop.
        
        max_steps : int
            Maximum number of steps to allow in the prototype-following process per input.

        allow_self_references : bool, optional (default=True)
            If False, the model will not allow a prototype to select itself as the next closest prototype.

        verbose : bool, optional (default=False)
            If True, prints detailed progress information

        Attributes Set
        --------------
        self.correctly_classified_array : np.ndarray
            Binary array indicating whether each input was initially classified correctly.

        self.class_preserv_broken_array : np.ndarray
            Binary array indicating if class preservation was broken during prototype traversal.

        self.cycle_found_array : np.ndarray
            Binary array indicating whether a prototype cycle was detected for each input.

        self.cycle_period_array : np.ndarray
            Integer array storing the length (period) of the detected cycle for each input.

        self.steps_converge_array : np.ndarray
            Number of steps taken until the cycle begins.

        self.steps_path_before_cycle_array : np.ndarray
            Number of steps taken before entering the cycle (i.e., the path length before repetition begins).

        Notes
        -----
        - This method assumes that the following abstract methods are implemented:
        - get_input_and_label()
        - get_prediction_and_distance()
        - get_pred_class()
        - get_prototype_at_idx()
        - get_cycles_from_prototypes()
        - get_cycle_path()
        - copy_input()
        """


        # Number of inputs
        self.num_inputs = num_inputs
        # Initialize array to store the resuñts
        self.correctly_classified_array, \
        self.class_preserv_broken_array, \
        self.cycle_found_array, \
        self.cycle_period_array, \
        self.steps_converge_array, \
        self.steps_path_before_cycle_array = [np.zeros(self.num_inputs, dtype=np.int32) for _ in range(6)]
                
        # Precompute cycles (just for sanity checks, but the overhead is negligible)
        converge_path_proto_precomputed, cycle_proto_precomputed = self.get_cycles_from_prototypes(allow_self_references)

        for idx in range(self.num_inputs):

            # Get the original input and label
            orig_input, orig_label_int = self.get_input_and_label(idx)

            # Visualize the original input
            if verbose:
                print("Input")
                self.show_input(orig_input)

            # Get the prediction and prototype-distances for the original input
            orig_logits, orig_proto_dist = self.get_prediction_and_distance(orig_input)
            orig_pred_int = self.get_pred_class(orig_logits) # Get original prediction (int)

            # Check if the prediction is correct, otherwise do not process the input
            if orig_pred_int == orig_label_int:
                self.correctly_classified_array[idx] = 1
            else:
                if verbose:
                    print(f"Incorrect prediction: True={orig_label_int} / Predicted={orig_pred_int}")
                continue

            # Get the order of the prototypes for the original prediction
            orig_order_idx = np.argsort(orig_proto_dist)
            if verbose:
                print("Original explanation")
                self.plot_prototypes(orig_order_idx, orig_proto_dist)

            # Initialize visited-prototype and next-prototype lists to check cycles
            visited_list  = {i:None for i in range(self.n_prototypes)}
            next_list     = {i:None for i in range(self.n_prototypes)}
            converge_path = []
            # Mark the closest prototype in the original prediction as visited
            next_closest_idx = orig_order_idx[0]
            visited_list[next_closest_idx] = 0
            converge_path.append(next_closest_idx)

            # Initialize variables needed for the iteration
            cur_input = self.copy_input(self.get_prototype_at_idx(next_closest_idx))
            cur_closest_idx = next_closest_idx

            #Initialize flags
            cycle_found_flag = False
            class_preserv_flag = False
            cycle_start_idx = None

            # Recursive steps
            for step in range(1, max_steps):
                if verbose:
                    print(f"Step {step}")
                    self.show_input(cur_input) #visualize current input

                # Get logits and similarities for the current input
                cur_logits, cur_proto_dist = self.get_prediction_and_distance(cur_input)
                if not allow_self_references:
                    cur_proto_dist[cur_closest_idx] = np.inf #disable the previous closest prototype
                cur_pred_int = self.get_pred_class(cur_logits) #get the class (int) for the current input

                # Check if class preservation has been broken
                if cur_pred_int != orig_label_int and not class_preserv_flag:
                    if verbose:
                        print(f"> Class preservation broken at step {step}")
                        print(f">> Incorrect prediction: True={orig_label_int} / Predicted={cur_pred_int}")
                    self.class_preserv_broken_array[idx] = 1 #class preservation is not satisfied for this input
                    class_preserv_flag = True #flag the class preservation so that it does not repeat this process again

                # Get the prototype order for the current input
                order_idx = np.argsort(cur_proto_dist)
                if verbose:
                    self.plot_prototypes(order_idx, cur_proto_dist)

                # Get the index of the closest prototype for the current input
                next_closest_idx = order_idx[0]

                # If the closest prototype has already been visited/process, flag that a cycle has been found!
                if visited_list[next_closest_idx] is not None:
                    if verbose: 
                        print(f"Cycle found for {idx}")
                    cycle_found_flag = True
                    cycle_start_idx = next_closest_idx
                    # Store the next-prototype for the previous index
                    next_list[cur_closest_idx] = next_closest_idx
                    # Store the number of steps needed until convergence
                    self.steps_converge_array[idx] = step
                    # Store the number of steps needed before the cycle starts
                    self.steps_path_before_cycle_array[idx] = copy.copy(visited_list[next_closest_idx])
                    break

                # If a cycle has not been found, set the next iteration
                visited_list[next_closest_idx] = step #mark in which step this index has been found
                next_list[cur_closest_idx] = next_closest_idx #fill the next-prototype list
                converge_path.append(next_closest_idx) #traversed path
                cur_input = self.copy_input(self.get_prototype_at_idx(next_closest_idx)) #next input will be the current prototype
                cur_closest_idx = next_closest_idx #update the previous index with the current one

            if not cycle_found_flag:
                print(f"No cycle found for {idx}")
            else:
                self.cycle_found_array[idx] = 1 #flag that a cycle has been found

                # Retrieve the cycle-path
                cycle_path = self.get_cycle_path(next_list, cycle_start_idx)
                # Get the period (length) of the cycle
                self.cycle_period_array[idx] = len(cycle_path)
                # Sanity check
                assert len(cycle_path)==len(converge_path[visited_list[cycle_start_idx]:])

                if verbose:
                    print(f"Full path:", converge_path)
                    print(f"Cycle path:", cycle_path)
                    print(f"Cycle period:", self.cycle_period_array[idx])
                    # Visualize the prototype-cycle
                    if len(cycle_path)==1: self.show_input(self.get_prototype_at_idx(cycle_path[0]))
                    else: self.plot_prototypes(cycle_path)
                    # Visualize the original explanation, for comparison
                    self.plot_prototypes(orig_order_idx, orig_proto_dist)

                # Sanity checks
                assert cycle_path==cycle_proto_precomputed[orig_order_idx[0]]
                assert converge_path==converge_path_proto_precomputed[orig_order_idx[0]]


    def get_cycle_path(self, next_list, cycle_start_idx):
        """ Auxiliary function to retrieve cycles

        Args:
            next_list (list): list pointing to the next prototype for each index (see evaluate) 
            cycle_start_idx (int): index at which the cycle starts

        Returns:
            list: list of indices forming the cycle
        """
        cycle_path = [cycle_start_idx]
        cur_idx = cycle_start_idx
        while next_list[cur_idx] != cycle_start_idx:
            cur_idx = next_list[cur_idx]
            cycle_path.append(cur_idx)
        return cycle_path


    def print_stats(self):
        """ 
        Auxiliary function to print stats of the recursion process. To be executed after evaluate()
        """
 
        mask = self.correctly_classified_array==1
        mask_sum = np.sum(mask)
        assert np.all(self.cycle_found_array[mask]==1)

        total_correct_classified_perc = np.round(mask_sum/self.num_inputs*100, 1)
        print(f"Correctly classified: {mask_sum}/{self.num_inputs} ({total_correct_classified_perc}%)")

        total_preserv_broken = np.sum(self.class_preserv_broken_array[mask])
        total_preserv_broken_perc = np.round(total_preserv_broken/mask_sum*100, 1)
        print(f"Class preservation broken: {total_preserv_broken}/{mask_sum} ({total_preserv_broken_perc}%)")
        print(f"Class preservation satisf: {100-total_preserv_broken_perc}%")

        # avg_period = np.mean(self.cycle_period_array[mask])
        # min_period = np.min(self.cycle_period_array[mask])
        # max_period = np.max(self.cycle_period_array[mask])
        # print(f"Cycle periods: min/avg/max: {min_period}/{avg_period.round(1)}/{max_period}")

        avg_step_conv = np.mean(self.steps_converge_array[mask])
        min_step_conv = np.min(self.steps_converge_array[mask])
        max_step_conv = np.max(self.steps_converge_array[mask])
        print(f"Steps to converge: min/avg/max: {min_step_conv}/{avg_step_conv.round(1)}/{max_step_conv}")

    
    def get_next_list(self):
        """
        Compute the closest prototype for each prototype in the model.

        Returns
        -------
        list of int
            A list `next_list` such that `next_list[i] = j` means the j-th prototype is 
            the closest (most similar) to the i-th prototype.
        """

        # Get logits and similarities for the current prototype-set
        prototypes = self.get_prototypes()
        _, cur_proto_dist = self.get_prediction_and_distance(prototypes)

        # Order, for each prototype, the prototypes by their distance 
        order_idx = np.argsort(cur_proto_dist, axis=1)
        # Get the closest prototypes
        next_list = np.copy(order_idx[:,0])
        return next_list


    def get_cycles_from_prototypes(self, allow_self_references=True, verbose=False):
        
        prototypes = self.get_prototypes()
        cur_logits, cur_proto_dist = self.get_prediction_and_distance(prototypes)

        converge_path = [[] for _ in range(self.n_prototypes)]
        cycle_proto   = [[] for _ in range(self.n_prototypes)]
        pre_cycle_part   = [[] for _ in range(self.n_prototypes)]
        for i in range(self.n_prototypes):
            cycle_found=False
            visited_idx_list_tmp = [0 for _ in range(self.n_prototypes)] 
            cur_idx = i
            while not cycle_found:
                # Mark prototype index as visited
                visited_idx_list_tmp[cur_idx] = 1
                converge_path[i].append(cur_idx)

                # Select next prototype (index)
                dists_cur = cur_proto_dist[cur_idx].copy() #get distances for current prototype
                if not allow_self_references:
                    dists_cur[cur_idx] = np.inf # Avoid previous prototype if discard flag is on
                next_idx = np.argmin(dists_cur) # Get the closest prototype

                if visited_idx_list_tmp[next_idx]==1:
                    cycle_found = True
                    #Retrieve the cycle only
                    cycle_start_pos = converge_path[i].index(next_idx) # first appearance of next_idx 
                    cycle_proto[i] = converge_path[i][cycle_start_pos:].copy() #only the cycle path
                    pre_cycle_part[i] = converge_path[i][:cycle_start_pos].copy() #only the cycle path
                else:
                    cur_idx  = next_idx
            if verbose:
                print(converge_path[i], next_idx, pre_cycle_part[i], cycle_proto[i])
        
        return converge_path, cycle_proto


    def plot_self_consistency(self, next_list, order=None, savepath=None, show=False):
        """ Plot the self-consistency of prototype-based models """

        # Assume prototype_imgs: (n_prototypes, C, H, W)
        prototype_imgs = self.get_prototypes()
        n_prototypes = prototype_imgs.shape[0]
        assert prototype_imgs.shape[0]==self.n_prototypes

        if isinstance(prototype_imgs, np.ndarray):
            proto_imgs_np = np.copy(prototype_imgs)
            proto_imgs_np = np.reshape(proto_imgs_np, (-1,28,28))
        if isinstance(prototype_imgs, torch.Tensor):
            proto_imgs_np = prototype_imgs.detach().cpu().numpy()
            proto_imgs_np = np.transpose(proto_imgs_np, (0,2,3,1))
            proto_imgs_np = np.squeeze(proto_imgs_np, axis=-1)

        if order is not None:
            # Reorder prototype images
            proto_imgs_np = proto_imgs_np[order]
            # Remap next_list based on new order
            reverse_order = np.argsort(order)
            next_list = reverse_order[next_list[order]]

        figsize = (18, 2.5) #(n_prototypes-20, 2.5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, n_prototypes)
        ax.set_ylim(0, 3)
        ax.axis('off')

        # Where to place the prototype images
        img_y_bottom = 0.5
        img_y_top = 1.5
        arrow_base_y = 1.6  # just above images

        # Plot prototype images in a row
        cmap = 'gray'
        for i in range(n_prototypes):
            img = proto_imgs_np[i]
            # if img.shape[0] == 1:
            #     img = img[0]
            # else:
            #     #img = np.transpose(img, (1, 2, 0))
            #     cmap = None

            ax.imshow(img, extent=(i + 0.0, i + 0.92, img_y_bottom, img_y_top), cmap=cmap)

        # Draw upward arching arrows for all i  j
        # Draw curved arrows
        for i, j in enumerate(next_list):
            if i == j:
                continue  # Skip self-links

            x_start = i + 0.5
            x_end = j + 0.5
            dist = abs(j - i)
            curve = 0.2 + 0.05 * dist  # base curvature magnitude

            # LEFT to RIGHT -> draw above, arch upward (∧)
            if j > i:
                y = 1.6  # arrow above images
                rad = -curve  # arch upward
            else:
                y = 1.6  # arrow below images
                rad = curve  # arch downward

            arrow = mpatches.FancyArrowPatch(
                (x_start, y), (x_end, y),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-|>',
                color='black',
                linewidth=1,
                mutation_scale=20
            )
            ax.add_patch(arrow)


        # Draw self-pointing arrows (loop below the prototype)
        for i, j in enumerate(next_list):
            if i != j:
                continue  # Only draw for self-links

            x = i + 0.5
            y = 0.5  # Below the image row
            loop_radius = 2

            arrow = mpatches.FancyArrowPatch(
                (x - 0.25, y), (x + 0.20, y),
                connectionstyle=f"arc3,rad={loop_radius}",
                arrowstyle='-|>',
                color='black',
                linewidth=1,
                mutation_scale=20
            )
            ax.add_patch(arrow)


        plt.tight_layout()
        if savepath is not None: plt.savefig(savepath, dpi=300)
        plt.show() if show else plt.close()


    def check_prototype_class_preservation(self, discard_previous=True, verbose=False):
        """
        Evaluates whether prototype-based recursive explanations preserve class labels
        during traversal, checking all possible recursion paths.

        Args:
            discard_previous (bool): Whether to discard the previous prototype at each step.
        """
        # Get recursion paths and cycles
        allow_self_references = (not discard_previous)
        paths, cycles = self.get_cycles_from_prototypes(allow_self_references)

        # Get class predictions for all prototypes
        prototypes = self.get_prototypes()
        logits, distances  = self.get_prediction_and_distance(prototypes)
        proto_classes = [self.get_pred_class(logits[i]) for i in range(self.n_prototypes)]

        # Track metrics
        class_preserved_flags = []

        for i, path in enumerate(paths):
            original_class = copy.copy(proto_classes[path[0]])
            preserved = all(proto_classes[j] == original_class for j in path)
            class_preserved_flags.append(preserved)
            if verbose:
                classes_path = [proto_classes[j] for j in path]
                print(f"{i}) orig-class: {original_class}, path-classes: {classes_path}")
                for j in path:
                    print(f"idx {j}) class {proto_classes[j]}")
                    self.show_input(prototypes[j])
                    self.plot_prototypes(np.argsort(distances[j]))

            cycle_classes = [proto_classes[j] for j in cycles[i]]
            pure_cycle = len(set(cycle_classes)) <= 1

        class_preserved_flags  = np.array(class_preserved_flags)

        # Reporting
        n = self.n_prototypes
        mean_class_preserved = np.mean(class_preserved_flags)
        print(f"Class preservation rate [allow_self_refs={not discard_previous}]: {np.sum(class_preserved_flags)}/{n} ({mean_class_preserved:.2%})")
        return mean_class_preserved

        


