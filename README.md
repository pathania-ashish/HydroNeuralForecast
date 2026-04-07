<div align="center">

# 🌊 HydroNeuralForecast

**Modified [Nixtla NeuralForecast](https://github.com/Nixtla/neuralforecast) (v3.0.2) for Hydrological Flood Detection**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](./Dockerfile)

</div>

---

> 🔬 This repository adapts Nixtla's NeuralForecast library for hydrological flood detection research. Key modifications include **multivariable support for models that originally lacked it** in the upstream repository, along with custom run scripts for exogenous variable handling.

---

## 🔧 What's Modified

### ⚙️ Enhanced Model Architectures

The following models have been modified to add **multivariable forecasting capabilities** that were **not available in the original NeuralForecast repository**. This enables these models to ingest multiple input variables (e.g., precipitation, temperature, soil moisture) simultaneously for improved flood prediction:

| Model | File | Modification |
|---|---|---|
| 🤖 **Informer** | `neuralforecast/models/informer.py` | Added multivariable support for hydrological inputs |
| 📊 **PatchTST** | `neuralforecast/models/patchtst.py` | Added multivariable support for hydrological inputs |

### 📜 Custom Model Run Scripts

Ready-to-use scripts for running models with exogenous and multivariate configurations:

| Script | Model | Description |
|---|---|---|
| 📄 `informer_aa_exog.py` | Informer | With exogenous variables |
| 📄 `LSTM_aa_exog.py` | LSTM | With exogenous variables |
| 📄 `PatchTST_aa_exog.py` | PatchTST | With exogenous variables |
| 📄 `KAN_multi_exag.py` | KAN | Multivariate with exogenous |
| 📄 `nhits_aa_multi.py` | NHITS | Multivariate |
| 📄 `tft_aa_exog.py` | TFT | With exogenous variables |

---

## 🚀 Setup & Installation

### 🐳 Option 1: Docker (Recommended)

Build and run the Docker container which replicates the `neuralforecast_3.10` conda environment:

```bash
git clone https://github.com/pathania-ashish/HydroNeuralForecast.git
cd HydroNeuralForecast

# Build the image
docker build -t hydroneuralforecast .

# Run interactively (with GPU support)
docker run --gpus all -it hydroneuralforecast bash
```

### 🐍 Option 2: Conda + pip

```bash
git clone https://github.com/pathania-ashish/HydroNeuralForecast.git
cd HydroNeuralForecast

# Create conda env with Python 3.10 and PyTorch (CUDA 12.4)
conda create -n neuralforecast_3.10 python=3.10 -y
conda activate neuralforecast_3.10
conda install pytorch=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install pip dependencies
pip install -r requirements.txt

# Install this package in editable mode
pip install -e .
```

---

## 📁 Project Structure

```
HydroNeuralForecast/
├── 📂 neuralforecast/
│   ├── 📂 models/            # All model architectures (modified: informer.py, patchtst.py)
│   ├── 📂 losses/            # Loss functions
│   ├── 📂 common/            # Base classes, modules, scalers
│   ├── 📄 core.py            # NeuralForecast core engine
│   ├── 📄 tsdataset.py       # Time series dataset handler
│   ├── 📄 auto.py            # Auto model selection
│   ├── 📄 informer_aa_exog.py
│   ├── 📄 LSTM_aa_exog.py
│   ├── 📄 PatchTST_aa_exog.py
│   ├── 📄 KAN_multi_exag.py
│   ├── 📄 nhits_aa_multi.py
│   └── 📄 tft_aa_exog.py
├── 🐳 Dockerfile             # Reproduces neuralforecast_3.10 environment
├── 📋 requirements.txt       # Pinned pip dependencies
├── ⚙️ setup.py
└── ⚙️ pyproject.toml
```

---

## 🙏 Acknowledgements

This project is built on top of [Nixtla's NeuralForecast](https://github.com/Nixtla/neuralforecast). Full credit to the original authors:

```bibtex
@misc{olivares2022library_neuralforecast,
    author={Kin G. Olivares and
            Cristian Challu and
            Azul Garza and
            Max Mergenthaler Canseco and
            Artur Dubrawski},
    title = {{NeuralForecast}: User friendly state-of-the-art neural forecasting models.},
    year={2022},
    howpublished={{PyCon} Salt Lake City, Utah, US 2022},
    url={https://github.com/Nixtla/neuralforecast}
}
```

## 📝 License

Apache 2.0. See [LICENSE](./LICENSE).
