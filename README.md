<img src="docs/images/replay_logo_color.svg" height="50"/>
<br>

This repository is a fork of the RePlay library, containing implementation Cut Cross Entropy (CCE) and Cut Cross Entropy with Negative Sampling (CCE-) for RecSys. Triton kernels are available in 
`kernels/cut_cross_entropy`. Implementation of SASRec with CCE and CCE- can be found in `replay/models/nn/sequential/sasrec/lightning.py`. Experiment pipeline is located in `replay_benchmarks`.

<a name="installation"></a>
## Installation

Installation via `poetry` package manager is recommended by default:

```bash
pip install --no-cache-dir --upgrade pip wheel poetry==1.5.1 poetry-dynamic-versioning \
    && python -m poetry config virtualenvs.create false
./poetry_wrapper.sh install --all-extras

```
After installing replay, it is required to update torch and install additional packages:
```bash
pip install --upgrade torch==2.5.1
pip install rs_datasets
pip install -U tensorboard
```

<a name="examples"></a>
## Usage

To run the experiments for training SASRec, use the following command from the RePlay-Accelerated directory:
```bash
python main.py
```

Experiment parameters are defined in `.yaml` files located in the configs directory. 
The dataset name is specified in the `config.yaml` file as follows:

Parameters for the experiments are defined by `.yaml` files in `configs` directory.
Name of the dataset in determined in the file `config.yaml`:
```
defaults:
  - dataset: <dataset_name>
  - model: sasrec_<dataset_name>
```
The following datasets are available `movielens_20m`, `beauty`, `30music`, `zvuk`, `megamarket`. 

Parameters for SASRec are defined in the sasrec_<dataset_name>.yaml files. 
To use CCE-, specify the following configuration:
```
loss_type: CCE
loss_sample_count: <number_of_negative_samples>
```
If `loss_sample_count: null`, the training will use the standard CCE loss.

To reproduce CE- grid search results, we provide a special trainer. It is available in `replay_benchmarks/grid_params_search_runner.py`. To set a grid for grid-search, you can modify the `replay_benchmarks/configs/mode/hyperparameter_experiment.yaml` file. Additionally, you need to change the usage mode in the main config (`replay_benchmarks/configs/config.yaml`). There, the parameter `mode: train` should be changed to `mode: hyperparameter_experiment`.

The `hyperparameter_experiment.yaml` configuration is used solely to iterate over `batch_size`, `max_seq_len`, and `loss_sample_count`. To change other parameters, you need to modify them in their respective configuration files.

## Acknowledgements 
This repository is build upon the [RePlay repository]
(https://github.com/sb-ai-lab/RePlay/tree/main). Triton kernels is based on the code of [ml-cross-entropy](
https://github.com/apple/ml-cross-entropy/tree/main).

