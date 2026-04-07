# HydroNeuralForecast

> **This repository is a modification of [Nixtla's NeuralForecast](https://github.com/Nixtla/neuralforecast) (v3.0.2).** The original library has been adapted for hydrological flood detection research, with custom modifications to model architectures and additional scripts for running models with exogenous variables.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

---

## What's Modified

### Modified Model Architectures
- **Informer** (`neuralforecast/models/informer.py`) — customized for hydrology use case
- **PatchTST** (`neuralforecast/models/patchtst.py`) — customized for hydrology use case

### Custom Model Run Scripts
| Script | Model | Description |
|---|---|---|
| `neuralforecast/informer_aa_exog.py` | Informer | With exogenous variables |
| `neuralforecast/LSTM_aa_exog.py` | LSTM | With exogenous variables |
| `neuralforecast/PatchTST_aa_exog.py` | PatchTST | With exogenous variables |
| `neuralforecast/KAN_multi_exag.py` | KAN | Multivariate with exogenous |
| `neuralforecast/nhits_aa_multi.py` | NHITS | Multivariate |
| `neuralforecast/tft_aa_exog.py` | TFT | With exogenous variables |

---

## Setup & Installation

### Option 1: Docker (Recommended)

Build and run the Docker container which replicates the `neuralforecast_3.10` conda environment:

```bash
git clone https://github.com/pathania-ashish/HydroNeuralForecast.git
cd HydroNeuralForecast

# Build the image
docker build -t hydroneuralforecast .

# Run interactively
docker run --gpus all -it hydroneuralforecast bash
```

### Option 2: Conda + pip

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

## Project Structure

```
HydroNeuralForecast/
├── neuralforecast/
│   ├── models/            # All model architectures (modified: informer.py, patchtst.py)
│   ├── losses/            # Loss functions
│   ├── common/            # Base classes, modules, scalers
│   ├── core.py            # NeuralForecast core engine
│   ├── tsdataset.py       # Time series dataset handler
│   ├── auto.py            # Auto model selection
│   ├── informer_aa_exog.py    # Custom run scripts
│   ├── LSTM_aa_exog.py
│   ├── PatchTST_aa_exog.py
│   ├── KAN_multi_exag.py
│   ├── nhits_aa_multi.py
│   └── tft_aa_exog.py
├── Dockerfile             # Reproduces neuralforecast_3.10 environment
├── requirements.txt       # Pinned pip dependencies
├── setup.py
└── pyproject.toml
```

---

## Acknowledgements

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

## License

Apache 2.0 (inherited from NeuralForecast). See [LICENSE](./LICENSE).
