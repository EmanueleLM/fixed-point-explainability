# Setup

Execute the following instructions:

```
cd ./prototype-based
sh setup.sh
```


# ProtoVAE

Our implementation is based on the [original implementation]((https://github.com/SrishtiGautam/ProtoVAE)) of [Gautam et al., 2022)](https://openreview.net/pdf?id=L8pZq2eRWvX). 


## Train

From the root directory of the project, execute:

```
cd ./prototype-based
python3 launch/train_protovae.py
```

The following parameters can be configured the script:
- `dataset`: Options: `["mnist", "fmnist"]`
- `npr`: Number of prototypes (10, 20, 50, 100) 

## Eval

In order to reproduce the results from the experiments for a dataset and a fixed number of prototyes, execute:
```
# from ./prototype-based
python3 protovae/eval_protovae.py -d [dataset] -npr [number]
```

The default settings can also be configured in `protovae/settings.py`, following the structure of the original implementation.



# PrototypeDNN


Our code is based on the [original implementation](https://github.com/OscarcarLi/PrototypeDL) from [(Li et al., 2018)](https://dl.acm.org/doi/abs/10.5555/3504035.3504467).
The training and evaluation processes are analogous to those of the previos case:


## Train

From root project, execute:

```
cd ./prototype-based
python3 launch/train_prototypeDNN.py
```

The following parameters can be configured the script:
- `dataset`: Options: `["mnist", "fashion"]`
- `npr`: Number of prototypes (10, 20, 50, 100) 

## Eval

In order to reproduce the results from the paper (for a dataset and a fixed number of prototyes), execute:
```
# from ./prototype-based
python3 prototypeDNN/eval_prototypeDNN.py
```

The default settings can be modified in the script.
