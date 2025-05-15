# %%
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import matplotlib.pyplot as plt
import random
import input_data


# %%
def is_local_env():
    """Returns True if running in an interactive environment"""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except:
        return False
        
if is_local_env(): root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
else: root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from RecexpPrototype import Recexp

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# %%
# Main parameters
dataset_name = "mnist" # Options: mnist, fashion
n_prototypes = 50      # Number of prototypes


# Fixed parameters
exp_idx      = 0  
n_features   = 40 
n_classes    = 10



# %%
if is_local_env(): 
    os.chdir("../")
print("Working in", os.getcwd())

# %%
if dataset_name == "mnist":
    mnist = input_data.read_data_sets("/tmp/MNIST_data", one_hot=True, validation_size=6000)
elif dataset_name == "fashion":
    dataset_dir = "data/fashion/"
    mnist = input_data.read_data_sets(dataset_dir,   one_hot=True, validation_size=6000)
else:
    sys.exit(f"Dataset {dataset_name} not supported yet")

tf.set_random_seed(0)   
print(mnist.validation.images.shape)
print(mnist.test.labels.shape)
test_images, test_labels = mnist.test.images, mnist.test.labels


#tf.compat.v1.reset_default_graph()

model_dir   = 'results/prototypeDNN/%s_npr_%d_cae_%d/' % (dataset_name, n_prototypes, exp_idx)
model_name  = 'cae-1499.meta'

print("\t+ Loading graph: " + model_dir + model_name)
#with tf.Session() as sess:    
sess = tf.Session(config=config) #added config=config to fix the error "failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED"
saver = tf.train.import_meta_graph(model_dir + model_name)
saver.restore(sess, tf.train.latest_checkpoint(model_dir))
graph = tf.get_default_graph()
print("\t- Graph loaded!")

print("\t+ Restoring tensors from the model...")
input_tensor         = graph.get_tensor_by_name("X:0")
label_tensor         = graph.get_tensor_by_name("Y:0")
input_decoded_tensor = graph.get_tensor_by_name("X_decoded:0")
first_conv_output    = graph.get_tensor_by_name("Relu_1:0") #output of the encoder (1rst conv layer)
second_conv_output   = graph.get_tensor_by_name("Relu_2:0") #output of the encoder (2nd conv layer)
last_conv_output     = graph.get_tensor_by_name("Relu_3:0") #output of the encoder
feature_vec_tensor   = graph.get_tensor_by_name("feature_vectors:0") #flattened output of the encoder
resh_feature_vec_tensor = graph.get_tensor_by_name("Reshape:0")
prototypes_tensor    = graph.get_tensor_by_name("prototype_feature_vectors:0")
deconv_bs_tensor     = graph.get_tensor_by_name("deconv_batch_size:0")
proto_dist_tensor    = graph.get_tensor_by_name("prototype_distances:0")
feature_dist_tensor  = graph.get_tensor_by_name("feature_vector_distances:0")
last_layer_w_tensor  = graph.get_tensor_by_name("last_layer_w:0")
logits_tensor        = graph.get_tensor_by_name("logits:0")
softmax_tensor       = graph.get_tensor_by_name("probability_distribution:0")
class_error_tensor   = graph.get_tensor_by_name("class_error:0")
print("\t- Tensors restored! Initialization sufccesfully completed!")

# %%
tmp_bs, tmp_rfv = sess.run([deconv_bs_tensor, resh_feature_vec_tensor],
                           feed_dict={feature_vec_tensor: prototypes_tensor.eval(session=sess)})

# prototype_imgs_1 = sess.run(input_decoded_tensor,
#                           feed_dict={deconv_bs_tensor:       tmp_bs, 
#                                      resh_feature_vec_tensor: tmp_rfv})
prototype_imgs = sess.run(input_decoded_tensor,
                          feed_dict={feature_vec_tensor: prototypes_tensor.eval(session=sess)})

# list_of_protos_latent = sess.run(prototypes_tensor) #shape=[n_prototypes, n_features]
# cur_proto_list = []
# for i in range(n_prototypes):
#     cur_proto = sess.run(input_decoded_tensor,
#                          feed_dict={feature_vec_tensor:list_of_protos_latent[i].reshape(1,n_features)})[0]
#     cur_proto = np.array(cur_proto).flatten()
#     cur_proto_list.append(cur_proto)
#     #assert np.all(np.abs(cur_proto-prototype_imgs[i])<1e-6)


# %%
proto_logits, proto_softmax = sess.run([logits_tensor,softmax_tensor],
                                       {input_tensor: prototype_imgs.reshape(-1,28*28)})
proto_argmax = np.argmax(proto_softmax, axis=1)
proto_class_order = np.argsort(proto_argmax)

# %%
# Implement Recursive Explainer (abstract methods, see Recexp.py for details)
class TF1Recexp(Recexp):
    def get_input_and_label(self, idx):
        return np.copy(test_images[idx,:]), np.argmax(np.copy(test_labels[idx]))

    def get_prediction_and_distance(self, input_data):
        ndim = len(input_data.shape)
        if ndim==1:
            input_data = input_data.reshape(1, -1).copy()
        logits, softmax, dist = sess.run(
            [logits_tensor, softmax_tensor, proto_dist_tensor], 
            {input_tensor: input_data}
        )
        if ndim==1: return logits[0], dist[0]
        else:       return logits, dist

    def get_pred_class(self, logits):
        return np.argmax(logits)

    def copy_input(self, input):
        return np.copy(input)

    def get_prototypes(self):
        return np.copy(prototype_imgs)

    def get_prototype_at_idx(self, proto_idx):
        return np.copy(prototype_imgs[proto_idx])

    def show_input(self, image):
        plt.imshow(image.reshape(28,28), cmap="gray")
        plt.axis("off")
        plt.gcf().set_size_inches(1,1)
        plt.show()

    def plot_prototypes(self, order_idx, distances=None):
        nrow = 1
        ncol = len(order_idx)
        figsize = (19,5) if len(order_idx)==self.n_prototypes else (1,4)
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
        for i in range(len(order_idx)):
            ax = axs[i] if nrow==1 else axs[i//ncol,i%ncol]
            ax.imshow(prototype_imgs[order_idx[i],:].reshape(28,28), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if not (distances is None): axs[i].set_title(distances[order_idx[i]].round(2), y=0.95)
        plt.show()
        



# %%
# Instantiate the recursive explainer
recexp = TF1Recexp(n_prototypes)

# %%
recexp.plot_prototypes(np.arange(n_prototypes))

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

# %%
savepath = f"selfcons_prototypeDNN_{dataset_name}_npr_{n_prototypes}_idx_{exp_idx}.pdf"
custom_order = np.arange(n_prototypes)
recexp.plot_self_consistency(next_list_nodiscard, savepath=savepath, show=True)
recexp.plot_self_consistency(next_list_nodiscard, proto_class_order, savepath=savepath, show=True)


# %%
# Evaluate recursive explanations for the test set (ALLOWING self-references)
num_inputs = len(test_images)
max_steps  = n_prototypes*n_prototypes+10  #maximum number of iterations (recursions) for each input
verbose= False

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


