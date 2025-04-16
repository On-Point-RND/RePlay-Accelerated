## 🎨 SRC_PROJECT
This repository containing implementation Cut Cross Entropy (CCE) and Cut Cross Entropy with Negative Sampling (CCE-) for RecSys. Triton kernels are available in 
`kernels/cut_cross_entropy`. Implementation of SASRec with CCE and CCE- can be found in `src/models/nn/sequential/sasrec/lightning.py`. Experiment pipeline is located in `src_benchmarks`.

## 🚀 Getting Started

Installation via Docker is recommended by default:

```bash
docker build -t src_project .
docker run -it --gpus '"device=0"' src_project
```

### 🐳 Docker Setup for Project
This repository includes a Dockerfile for building a development environment with:
- PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
- Python 3.10 (from the PyTorch base image)
- Java 11 (OpenJDK)

### 📦 Installed System Packages (apt)
The following system dependencies are installed in the image:
- apt-utils – base utility for apt
- build-essential – compiler and toolchain (required by some Python packages)
- libgomp1 – OpenMP support library (used by numpy and other libs)
- pandoc – required for Lightning logs or documentation exports
- git – for cloning repos or versioning
- openjdk-11 – manually copied from the slim OpenJDK image

### 🧪 Python Packages (pip)
Installed Python packages with pinned versions:
```plaintext
  numpy==1.24.4
  lightning==2.5.1
  pandas==1.5.3
  polars==1.0.0
  optuna==3.2.0
  scipy==1.9.3
  psutil==6.0.0
  scikit-learn==1.3.2
  pyarrow==16.0.0
  torch==2.5.1
  rs_datasets==0.5.1
  Ninja==1.11.1.1
  tensorboard==2.19.0
```
## ⚙️ Running Experiments with SASRec
### ⚡️ Quickstart
To run the experiments for training SASRec, use the following command from the project directory:
```bash
python main.py
```
### 📁 Configuration
All experiment parameters are defined using .yaml configuration files located in the `src_benchmarks/configs` directory.
The main configuration file is `src_benchmarks/configs/config.yaml`, where you specify the dataset and model:

```yaml
defaults:
  - dataset: <dataset_name>
  - model: sasrec_<dataset_name>
```

**Available datasets:** 
- movielens_20m
- beauty
- 30music
- zvuk
- megamarket
- gowalla

Each dataset have a corresponding config file, e.g., `src_benchmarks/configs/dataset/movielens_20m.yaml`.

### ⚙️ Custom Loss Configuration
To use CCE- loss, include the following parameters in the model config (sasrec_<dataset_name>.yaml):
```yaml
- loss_type: CCE
- loss_sample_count: <number_of_negative_samples>
```
If `loss_sample_count: null`, the training will use the standard CCE method.

### 🔍 Reproducing CE- Grid Search
We provide a dedicated trainer for grid search, located at:
```bash
src_benchmarks/grid_params_search_runner.py
```
To configure a hyperparameter grid:
1. Modify the file:
`src_benchmarks/configs/mode/hyperparameter_experiment.yaml`
(This controls the grid over `batch_size`, `max_seq_len`, and `loss_sample_count`.)
2. Set the mode to "hyperparameter_experiment" in:
`src_benchmarks/configs/config.yaml`

```yaml
defaults:
  ...
  - mode: hyperparameter_experiment
```

> 💡**P.S.** To adjust other training parameters, edit them in their respective config files (e.g., `src_benchmarks/configs/model/sasrec_movielens_20m.yaml`), not in `hyperparameter_experiment.yaml`.


