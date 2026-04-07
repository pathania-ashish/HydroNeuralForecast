# HydroNeuralForecast

> **This repository is a modification of [Nixtla's NeuralForecast](https://github.com/Nixtla/neuralforecast).** The original library has been adapted for hydrological research, with custom modifications to model architectures (Informer, PatchTST) and additional model run scripts for exogenous variable support.

---

<div align="center">
<h1 align="center">HydroNeuralForecast</h1>
<h3 align="center">Modified NeuralForecast for Hydrology & Flood Detection</h3>

[![CI](https://github.com/Nixtla/neuralforecast/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/neuralforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/neuralforecast)](https://pypi.org/project/neuralforecast/)
[![PyPi](https://img.shields.io/pypi/v/neuralforecast?color=blue)](https://pypi.org/project/neuralforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/conda-forge/neuralforecast?color=seagreen&label=conda)](https://anaconda.org/conda-forge/neuralforecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/neuralforecast.svg?label=docs)](https://nixtla.github.io/neuralforecast/)  
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-11-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

**NeuralForecast** offers a large collection of neural forecasting models focusing on their performance, usability, and robustness. The models range from classic networks like RNNs to the latest transformers: `MLP`, `LSTM`, `GRU`, `RNN`, `TCN`, `TimesNet`, `BiTCN`, `DeepAR`, `NBEATS`, `NBEATSx`, `NHITS`, `TiDE`, `DeepNPTS`, `TSMixer`, `TSMixerx`, `MLPMultivariate`, `DLinear`, `NLinear`, `TFT`, `Informer`, `AutoFormer`, `FedFormer`, `PatchTST`, `iTransformer`, `StemGNN`, and `TimeLLM`.
</div>

## Installation

You can install `NeuralForecast` with:

```python
pip install neuralforecast
```

or 

```python
conda install -c conda-forge neuralforecast
``` 
Vist our [Installation Guide](https://nixtlaverse.nixtla.io/neuralforecast/docs/getting-started/installation.html) for further details.

## Quick Start

**Minimal Example**

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.utils import AirPassengersDF

nf = NeuralForecast(
    models = [NBEATS(input_size=24, h=12, max_steps=100)],
    freq = 'ME'
)

nf.fit(df=AirPassengersDF)
nf.predict()
```

**Get Started with this [quick guide](https://nixtlaverse.nixtla.io/neuralforecast/docs/getting-started/quickstart.html).**

## Why? 

There is a shared belief in Neural forecasting methods' capacity to improve forecasting pipeline's accuracy and efficiency.

Unfortunately, available implementations and published research are yet to realize neural networks' potential. They are hard to use and continuously fail to improve over statistical methods while being computationally prohibitive. For this reason, we created `NeuralForecast`, a library favoring proven accurate and efficient models focusing on their usability.

## Features 

* Fast and accurate implementations of more than 30 state-of-the-art models. See the entire [collection here](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html).
* Support for exogenous variables and static covariates.
* Interpretability methods for trend, seasonality and exogenous components.
* Probabilistic Forecasting with adapters for quantile losses and parametric distributions.
* Train and Evaluation Losses with scale-dependent, percentage and scale independent errors, and parametric likelihoods.
* Automatic Model Selection with distributed automatic hyperparameter tuning.
* Familiar sklearn syntax: `.fit` and `.predict`.

## Highlights

* Official `NHITS` implementation, published at AAAI 2023. See [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25854) and [experiments](./experiments/).
* Official `NBEATSx` implementation, published at the International Journal of Forecasting. See [paper](https://www.sciencedirect.com/science/article/pii/S0169207022000413).
* Unified with`StatsForecast`, `MLForecast`, and `HierarchicalForecast` interface `NeuralForecast().fit(Y_df).predict()`, inputs and outputs.
* Built-in integrations with `utilsforecast` and `coreforecast` for visualization and data-wrangling efficient methods.
* Integrations with `Ray` and `Optuna` for automatic hyperparameter optimization.
* Predict with little to no history using Transfer learning. Check the experiments [here](https://github.com/Nixtla/transfer-learning-time-series).

Missing something? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## Examples and Guides

The [documentation page](https://nixtlaverse.nixtla.io/neuralforecast/docs/getting-started/introduction.html) contains all the examples and tutorials.

📈 [Automatic Hyperparameter Optimization](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/hyperparameter_tuning.html): Easy and Scalable Automatic Hyperparameter Optimization with `Auto` models on `Ray` or `Optuna`.

🌡️ [Exogenous Regressors](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html): How to incorporate static or temporal exogenous covariates like weather or prices.

🔌 [Transformer Models](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/longhorizon_transformers.html): Learn how to forecast with many state-of-the-art Transformers models.

👑 [Hierarchical Forecasting](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/hierarchical_forecasting.html): forecast series with very few non-zero observations. 

👩‍🔬 [Add Your Own Model](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/adding_models.html): Learn how to add a new model to the library.

## Models

See the entire [collection here](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html).

Missing a model? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## How to contribute
If you wish to contribute to the project, please refer to our [contribution guidelines](https://github.com/Nixtla/neuralforecast/blob/main/CONTRIBUTING.md).

## References
This work is highly influenced by the fantastic work of previous contributors and other scholars on the neural forecasting methods presented here. We want to highlight the work of [Boris Oreshkin](https://arxiv.org/abs/1905.10437), [Slawek Smyl](https://www.sciencedirect.com/science/article/pii/S0169207019301153), [Bryan Lim](https://www.sciencedirect.com/science/article/pii/S0169207021000637), and [David Salinas](https://arxiv.org/abs/1704.04110). We refer to [Benidis et al.](https://arxiv.org/abs/2004.10240) for a comprehensive survey of neural forecasting methods.





<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
