### Instructions 

There are 3 .ipynb files. Each of them allow to replicate all the experiments for the feauture based explainers, i.e., LIME, SHAP, and LRP.

For SHAP, there is a patch you need to apply first to save images. While we plan to release a simple patch soon, for now, once you installed shap via pip (it's done automatically),
locate ```_image.py``` inside of your environment. The path should be like this one: ```.env/lib/python3.10/site-packages/shap/plots/``` and remove ```_image.py``` with ```_image.py``` inside ```patch-shap``` folder.

While for LIME and SHAP there are pip packages installed previously, for LRP you need to clone this repository (https://github.com/kaifishr/PyTorchRelevancePropagation) inside the ```feature-based``` folder.
A ```git clone https://github.com/kaifishr/PyTorchRelevancePropagation``` should be sufficient. You should now have a folder named ```PyTorchRelevancePropagation``` inside of ```feature-based``` folder.

Then, you can just run each of the .ipynb notebook using the previously created environment as a "kernel".